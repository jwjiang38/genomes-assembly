#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <numeric>
#include <set>
#include <upcxx/upcxx.hpp>
#include <vector>

#include "hash_map.hpp"
#include "kmer_t.hpp"
#include "read_kmers.hpp"

#include "butil.hpp"
//what's new: update gindex by nbucket, faster inserting
using namespace::std;
uint64_t expfac = 2;
//GASNET_MAX_SEGSIZE = ;
int main(int argc, char** argv) {
    upcxx::init();
    int rankme = upcxx::rank_me();
    int rankn = upcxx::rank_n();

    // TODO: Dear Students,
    // Please remove this if statement, when you start writing your parallel implementation.
 //   if (upcxx::rank_n() > 1) {
 //       throw std::runtime_error("Error: parallel implementation not started yet!"
 //                                " (remove this when you start working.)");
 //   }

    if (argc < 2) {
        BUtil::print("usage: srun -N nodes -n ranks ./kmer_hash kmer_file [verbose|test]\n");
        upcxx::finalize();
        exit(1);
    }

    std::string kmer_fname = std::string(argv[1]);
    std::string run_type = "";

    if (argc >= 3) {
        run_type = std::string(argv[2]);
    }

    int ks = kmer_size(kmer_fname);

    if (ks != KMER_LEN) {
        throw std::runtime_error("Error: " + kmer_fname + " contains " + std::to_string(ks) +
                                 "-mers, while this binary is compiled for " +
                                 std::to_string(KMER_LEN) +
                                 "-mers.  Modify packing.hpp and recompile.");
    }

    size_t n_kmers = line_count(kmer_fname);

    // Load factor of 0.5
//    size_t hash_table_size = n_kmers * (1.0 / 0.5);
//    HashMap hashmap(hash_table_size);

    //if (run_type == "verbose") {
    //    BUtil::print("Initializing hash table of size %d for %d kmers.\n", hash_table_size,
    //                 n_kmers);
    //}

    std::vector<kmer_pair> kmers = read_kmers(kmer_fname, rankn, rankme);
    uint64_t localsize = ceil(n_kmers/rankn)*expfac;
    if (run_type == "verbose") {
        BUtil::print("Finished reading kmers.\n");
    }
   // cout<<"rank "<<upcxx::rank_me()<<"has "<<localsize/expfac<<" kmers of total "<<n_kmers<<endl;
   // cout.flush();
    upcxx::global_ptr<kmer_pair> kmerdat[rankn];
    upcxx::global_ptr<int> kmern[rankn];
    kmerdat[rankme]=upcxx::new_array<kmer_pair>(localsize);
    kmern[rankme]=upcxx::new_array<int>(localsize);
    for (int i=0;i<rankn;i++){
        kmerdat[i]=upcxx::broadcast(kmerdat[i],i).wait();
        kmern[i]=upcxx::broadcast(kmern[i],i).wait();
    }
    upcxx::atomic_domain<int> ad({upcxx::atomic_op::load, upcxx::atomic_op::fetch_add});
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<kmer_pair*> start_nodes;
    for (int i=0;i<kmers.size();i++) {
        kmer_pair* kmer = &kmers[i];
        uint64_t hashval = kmer->hash(); 
	int trank = hashval%rankn;
	uint64_t gindex = hashval%(localsize);
        bool success=false;
	int count =0;
        while(!success){
            int nbucket = ad.fetch_add(kmern[trank]+gindex,1, memory_order_relaxed).wait();
            if (nbucket==0){
	        upcxx::rput(*kmer,kmerdat[trank]+gindex).wait();
		success = true;
		break;
	    }
	    gindex=(gindex+nbucket)%localsize;
	    count++;
	    if (count==localsize){
                throw std::runtime_error("Error: HashMap is full!");
	    }
	}
        //bool success = hashmap.insert(kmer);
        //if (!success) {
       //     throw std::runtime_error("Error: HashMap is full!");
       // }

        if (kmer->backwardExt() == 'F') {
            start_nodes.push_back(kmer);
        }
    }

    auto end_insert = std::chrono::high_resolution_clock::now();
    upcxx::barrier();

    double insert_time = std::chrono::duration<double>(end_insert - start).count();
    if (run_type != "test") {
        BUtil::print("Finished inserting in %lf\n", insert_time);
    }
    upcxx::barrier();

    auto start_read = std::chrono::high_resolution_clock::now();
    upcxx::barrier();
    std::list<kmer_pair> contigs[start_nodes.size()];
    int i=0;
    for (const auto& start_kmer : start_nodes) {
	std::list<kmer_pair>* contig = &contigs[i];
        contig->push_back(*start_kmer);
        while (contig->back().forwardExt() != 'F') {
            uint64_t hashval = contig->back().next_kmer().hash(); 
            int trank = hashval%rankn;
    	    uint64_t gindex = hashval%localsize;
	    bool success = false;
	    int count=0;
	    while(!success){
    	        kmer_pair getkmer = upcxx::rget(kmerdat[trank]+gindex).wait();
		if (getkmer.kmer == contig->back().next_kmer()){
                    success = true;
		    contig->push_back(getkmer);
		    break;
		}
		gindex=(gindex+1)%localsize;
		count++;
		if (count==localsize){
		    cout<<"can't find"<<contig->back().next_kmer().get()<<endl;
		    cout<<"trid to find it in rank "<<trank<<" index "<<gindex<<endl;
                    throw std::runtime_error("Error: kmer not found!");
		}
	    }
        }
        i++;
    }

    auto end_read = std::chrono::high_resolution_clock::now();
    upcxx::barrier();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> read = end_read - start_read;
    std::chrono::duration<double> insert = end_insert - start;
    std::chrono::duration<double> total = end - start;

    int numKmers=0;
    for (int i=0;i<start_nodes.size();i++){
        numKmers+=contigs[i].size();
    }


    if (run_type != "test") {
        BUtil::print("Assembled in %lf total\n", total.count());
    }

    if (run_type == "verbose") {
        printf("Rank %d reconstructed %d contigs with %d nodes from %d start nodes."
               " (%lf read, %lf insert, %lf total)\n",
               rankme, start_nodes.size(), numKmers, start_nodes.size(), read.count(),
               insert.count(), total.count());
    }

    if (run_type == "test") {
        std::ofstream fout("test_" + std::to_string(rankme) + ".dat");
        for (const auto& contig : contigs) {
            fout << extract_contig(contig) << std::endl;
        }
        fout.close();
    }

    upcxx::finalize();
    return 0;
}
