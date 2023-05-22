#include <omp.h>
#include <chrono>

#include <fabry/fabry.hpp>
#include <networkit/auxiliary/Random.hpp>
#include <networkit/io/NetworkitBinaryReader.hpp>

#include "DistributedKadabra.hpp"

int main(int argc, char **argv) {
    fabry::program p{&argc, &argv};
    fabry::communicator world{fabry::world};

    double epsilon = 0.001;
    unsigned int seed = 42;
    bool deterministic = false;

    if(world.is_rank_zero()) {
        std::cout << "node_bits: " << (sizeof(NetworKit::storednode) * 8) << std::endl;
        std::cout << "instance: '" << argv[1] << "'" << std::endl;
        std::cout << "seed: " << seed << std::endl;
        std::cout << "epsilon: " << epsilon << std::endl;
        std::cout << "deterministic: " << (int)deterministic << std::endl;
        std::cout << "num_procs: " << world.n_ranks() << std::endl;
        std::cout << "threads_per_proc: " << omp_get_max_threads() << std::endl;
    }

    Aux::Random::setSeed(seed, false);

    NetworKit::NetworkitBinaryReader reader;
    auto g = reader.read(argv[1]);
    if(g.isDirected())
        throw std::runtime_error("Graph needs to be undirected");
    if(g.isWeighted())
        throw std::runtime_error("Graph needs to be unweighted");

    NetworKit::DistributedKadabra algo{g, epsilon, 0.1, deterministic};

    // Do a barrier so that the algorithm can be timed correctly.
    fabry::post(world.barrier(fabry::collective));

    auto timeStart = std::chrono::high_resolution_clock::now();
    algo.run();
    auto durElapsed = std::chrono::high_resolution_clock::now() - timeStart;
    auto secElapsed = std::chrono::duration_cast<std::chrono::duration<double>>(durElapsed);

    if(world.is_rank_zero()) {
        std::cout << "num_nodes: " << g.upperNodeIdBound() << std::endl;
        std::cout << "num_edges: " << g.numberOfEdges() << std::endl;
        std::cout << "num_epochs: " << algo.numEpochs() << std::endl;
        std::cout << "num_samples: " << algo.getNumberOfIterations() << std::endl;
        std::cout << "time: " << secElapsed.count() << std::endl;
        std::cout << "mtime_diam: " << algo.diamTime << std::endl;
        std::cout << "mtime_phase1: " << algo.phase1Time << std::endl;
        std::cout << "mtime_estimate: " << algo.phase1EstimateTime << std::endl;
        std::cout << "mtime_phase2: " << algo.phase2Time << std::endl;
        std::cout << "mtime_sync: " << algo.phase2SyncTime << std::endl;
        std::cout << "mtime_transition: " << algo.phase2TransitionTime << std::endl;
        std::cout << "mtime_barrier: " << algo.phase2BarrierTime << std::endl;
        std::cout << "io_mtime_barrier: " << algo.phase2BarrierIoTime << std::endl;
        std::cout << "overlap_mtime_barrier: " << algo.phase2BarrierOverlapTime << std::endl;
        std::cout << "mtime_reduce: " << algo.phase2ReduceTime << std::endl;
        std::cout << "io_mtime_reduce: " << algo.phase2ReduceIoTime << std::endl;
        std::cout << "overlap_mtime_reduce: " << algo.phase2ReduceOverlapTime << std::endl;
        std::cout << "mtime_check: " << algo.phase2CheckTime << std::endl;
        std::cout << "mtime_bcast: " << algo.phase2BcastTime << std::endl;

        auto ranking = algo.ranking();
        std::cout << "ranking:" << std::endl;
        for(size_t i = 0; i < 10; i++) {
            if(i >= ranking.size())
                break;
            std::cout
                    << " -  rank: " << (i + 1) << "\n"
                    << "    node: " << ranking[i].first << "\n"
                    << "    score: " << ranking[i].second << std::endl;
        }
    }
}
