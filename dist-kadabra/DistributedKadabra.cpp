/*
 * DistributedKadabra.cpp
 *
 * Created on: 18.07.2018
 *    Authors: Eugenio Angriman <angrimae@hu-berlin.de>
 *             Alexander van der Grinten <avdgrinten@hu-berlin.de>
 */

#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <omp.h>

#include <fabry/fabry.hpp>
#include <networkit/auxiliary/Parallel.hpp>
#include <networkit/auxiliary/Parallelism.hpp>
#include <networkit/auxiliary/Random.hpp>
#include <networkit/auxiliary/Timer.hpp>
#include <networkit/distance/Diameter.hpp>

#include "DistributedKadabra.hpp"

namespace NetworKit {

DistributedStatus::DistributedStatus(const count k)
    : k(k), top(k), approxTop(k), finished(k), bet(k), errL(k), errU(k) {}

DistributedKadabra::DistributedKadabra(const Graph &G, const double err,
                                       const double delta,
                                       const bool deterministic, const count k,
                                       count unionSample,
                                       const count startFactor)
    : G(G), delta(delta), err(err), deterministic(deterministic), k(k),
      startFactor(startFactor), unionSample(unionSample), absolute(k == 0),
      stop(false) {
    const count n = G.upperNodeIdBound();
    if (k > n)
        throw std::runtime_error(
            "k is higher than the number of nodes of the input graph! Choose a "
            "value between 0 (absolute) and n-1.");

    if (delta >= 1 || delta <= 0)
        throw std::runtime_error(
            "Delta should be greater than 0 and smaller than 1.");

    if (err >= 1 || err <= 0)
        throw std::runtime_error(
            "The error should be greater than 0 and smaller than 1.");

    seed0 = Aux::Random::integer();
    seed1 = Aux::Random::integer();
}

bool DistributedKadabra::computeFinished(DistributedStatus *status) const {
    std::vector<double> &bet = status->bet;
    std::vector<double> &errL = status->errL;
    std::vector<double> &errU = status->errU;
    bool allFinished = true;

    count i;
    for (i = 0; i < status->k - 1; ++i) {
        bet[i] = status->approxTop[i] / (double)nPairs;
        errL[i] = computeF(bet[i], nPairs, deltaLGuess[status->top[i]]);
        errU[i] = computeG(bet[i], nPairs, deltaUGuess[status->top[i]]);
    }

    bet[i] = status->approxTop[i] / (double)nPairs;
    errL[i] = computeF(bet[i], nPairs, this->deltaLMinGuess);
    errU[i] = computeG(bet[i], nPairs, this->deltaUMinGuess);

    if (absolute) {
        for (count i = 0; i < status->k; ++i) {
            status->finished[i] = (errL[i] < err && errU[i] < err);
            allFinished = allFinished && status->finished[i];
        }
    } else {
        for (count i = 0; i < status->k; ++i) {
            if (i == 0) {
                status->finished[i] =
                    (bet[i] - errL[i] > bet[i + 1] + errU[i + 1]);
            } else if (i < k) {
                status->finished[i] =
                    (bet[i - 1] - errL[i - 1] > bet[i] + errU[i]) &&
                    (bet[i] - errL[i] > bet[i + 1] + errU[i + 1]);
            } else {
                status->finished[i] =
                    bet[k - 1] - errU[k - 1] > bet[i] + errU[i];
            }
            status->finished[i] =
                status->finished[i] || (errL[i] < err && errU[i] < err);
            allFinished = allFinished && status->finished[i];
        }
    }

    return allFinished;
}

// Computes the function f that bounds the betweenness of a vertex from below.
// For more information, see Borassi, Natale (2016).
double DistributedKadabra::computeF(const double btilde, const count iterNum,
                                    const double deltaL) const {
    double tmp = (((double)omega) / iterNum - 1. / 3);
    double errChern =
        (std::log(1. / deltaL)) * 1. / iterNum *
        (-tmp +
         std::sqrt(tmp * tmp + 2 * btilde * omega / (std::log(1. / deltaL))));
    return std::min(errChern, btilde);
}

// Computes the function g that bounds the betweenness of a vertex from above.
// For more information, see Borassi, Natale (2016).
double DistributedKadabra::computeG(const double btilde, const count iterNum,
                                    const double deltaU) const {
    double tmp = (((double)omega) / iterNum + 1. / 3);
    double errChern =
        (std::log(1. / deltaU)) * 1. / iterNum *
        (tmp +
         std::sqrt(tmp * tmp + 2 * btilde * omega / (std::log(1. / deltaU))));
    return std::min(errChern, 1 - btilde);
}

void DistributedKadabra::getStatus(DistributedStatus *status, const bool parallel) const {
    if (status != NULL) {
        auto loop = [&](count i) {
            if (absolute) {
                status->top[i] = i;
                status->approxTop[i] = approxSum[i];
            } else {
                status->top[i] = top->getElement(i);
                status->approxTop[i] = top->getValue(i);
            }
        };
        if (parallel) {
#pragma omp parallel for
            for (omp_index i = 0; i < static_cast<omp_index>(unionSample);
                 ++i) {
                loop(static_cast<count>(i));
            }
        } else {
            for (count i = 0; i < unionSample; ++i) {
                loop(i);
            }
        }
    }
}

void DistributedKadabra::computeBetErr(DistributedStatus *status, std::vector<double> &bet,
                                       std::vector<double> &errL,
                                       std::vector<double> &errU) const {
    count i;
    double maxErr = std::sqrt(startFactor) * err / 4.;

    for (i = 0; i < status->k; ++i)
        bet[i] = status->approxTop[i] / (double)nPairs;

    if (absolute) {
        for (i = 0; i < status->k; ++i) {
            errL[i] = err;
            errU[i] = err;
        }
    } else {
        errU[0] = std::max(err, (bet[0] - bet[1]) / 2.);
        errL[0] = 10;
        for (i = 1; i < k; ++i) {
            errL[i] = std::max(err, (bet[i - 1] - bet[i]) / 2.);
            errU[i] = std::max(err, (bet[i] - bet[i + 1]) / 2.);
        }
        for (i = k; i < status->k; ++i) {
            errL[i] = 10;
            errU[i] =
                std::max(err, bet[k - 1] + (bet[k - 1] - bet[k]) / 2. - bet[i]);
        }
        for (i = 0; i < k - 1; ++i) {
            if (bet[i] - bet[i + 1] < maxErr) {
                errL[i] = err;
                errU[i] = err;
                errL[i + 1] = err;
                errU[i + 1] = err;
            }
        }
        for (i = k + 1; i < status->k; ++i) {
            if (bet[k] - bet[i] < maxErr) {
                errL[k] = err;
                errU[k] = err;
                errL[i] = err;
                errU[i] = err;
            }
        }
    }
}

void DistributedKadabra::computeDeltaGuess() {
    const count n = G.upperNodeIdBound();
    const double balancingFactor = 0.001;
    double a = 0,
           b = 1. / err / err * std::log(n * 4 * (1 - balancingFactor) / delta),
           c = (a + b) / 2;
    double sum;

    DistributedStatus status(unionSample);
    getStatus(&status, true);

    std::vector<double> bet(status.k);
    std::vector<double> errL(status.k);
    std::vector<double> errU(status.k);
    std::vector<double> errLTerm(status.k);
    std::vector<double> errUTerm(status.k);

    computeBetErr(&status, bet, errL, errU);

#pragma omp parallel for
    for (count i = 0; i < unionSample; ++i) {
        count v = status.top[i];
        approxSum[v] = approxSum[v] / (double)nPairs;
        errLTerm[i] = errL[i] * errL[i] / bet[i];
        errUTerm[i] = errU[i] * errU[i] / bet[i];
    }

    int binSearchIters = 0;
    while (b - a > err / 10.) {
        c = (b + a) / 2.;
        sum = 0;
#pragma omp parallel for
        for (omp_index i = 0; i < static_cast<omp_index>(unionSample); ++i) {
            sum += std::exp(-c * errLTerm[i]);
            sum += std::exp(-c * errUTerm[i]);
        }

        sum += std::exp(-c * errLTerm[unionSample - 1]) * (n - unionSample);
        sum += std::exp(-c * errUTerm[unionSample - 1]) * (n - unionSample);

        if (sum >= delta / 2. * (1 - balancingFactor))
            a = c;
        else
            b = c;
        binSearchIters++;
    }
    INFO("iterations in binary search: ", binSearchIters);

    deltaLMinGuess = std::exp(-b * errL[unionSample - 1] *
                              errL[unionSample - 1] / bet[unionSample - 1]) +
                     delta * balancingFactor / 4. / (double)n;
    deltaUMinGuess = std::exp(-b * errU[unionSample - 1] *
                              errU[unionSample - 1] / bet[unionSample - 1]) +
                     delta * balancingFactor / 4. / (double)n;

#pragma omp parallel for
    for (omp_index i = 0; i < static_cast<omp_index>(unionSample); ++i) {
        node v = status.top[i];
        deltaLGuess[v] = std::exp(-b * errL[i] * errL[i] / bet[i]) +
                         delta * balancingFactor / 4. / (double)n;
        deltaUGuess[v] = std::exp(-b * errU[i] * errU[i] / bet[i]) +
                         delta * balancingFactor / 4. / (double)n;
    }
}

void DistributedKadabra::computeApproxParallel(
    const std::vector<DistributedStateFrame> &firstFrames) {
    const count omp_max_threads = omp_get_max_threads();
#pragma omp parallel for
    for (omp_index i = 0; i < static_cast<omp_index>(G.upperNodeIdBound());
         ++i) {
        for (count j = 0; j < omp_max_threads; ++j) {
            approxSum[i] += firstFrames[j].apx[i];
        }
    }
}

void DistributedKadabra::init() {
    const count n = G.upperNodeIdBound();
    const count omp_max_threads = omp_get_max_threads();
    approxSum.resize(n, 0);
    deltaLGuess.resize(n, 0);
    deltaUGuess.resize(n, 0);
    cc = std::unique_ptr<ConnectedComponents>(new ConnectedComponents(G));
    if (determineComponents && !G.isDirected())
        cc->run();
    epochFinished = std::vector<std::atomic<DistributedStateFrame *>>(omp_max_threads);
    samplerVec.reserve(omp_max_threads);
    for (count i = 0; i < omp_max_threads; ++i) {
        samplerVec.emplace_back(DistributedSpSampler(G, *cc));
        epochFinished[i].store(nullptr, std::memory_order_relaxed);
    }

    maxFrames.resize(omp_max_threads, 0);
}

void DistributedKadabra::fillResult() {
    const count n = G.upperNodeIdBound();
    if (absolute) {
        topkScores.resize(n);
        topkNodes.resize(n);
        rankingVector.resize(n);
#pragma omp parallel for
        for (omp_index i = 0; i < static_cast<omp_index>(n); ++i) {
            rankingVector[i] = std::make_pair(i, approxSum[i]);
        }
        Aux::Parallel::sort(
            rankingVector.begin(), rankingVector.end(),
            [&](std::pair<node, double> p1, std::pair<node, double> p2) {
                return p1.second > p2.second;
            });
#pragma omp parallel for
        for (omp_index i = 0; i < static_cast<omp_index>(n); ++i) {
            topkNodes[i] = rankingVector[i].first;
            topkScores[i] = rankingVector[i].second;
        }
    } else {
        topkScores.resize(k);
        topkNodes.resize(k);
        rankingVector.resize(k);
        for (count i = 0; i < k; ++i) {
            topkNodes[i] = top->getElement(i);
            topkScores[i] = approxSum[topkNodes[i]];
            assert(top->getValue(i) == topkScores[i]);
            rankingVector[i] = std::make_pair(topkNodes[i], topkScores[i]);
        }
    }
}

void DistributedKadabra::run() {
    init();
    fabry::communicator world{fabry::world};
    const count n = G.upperNodeIdBound();
    const auto omp_max_threads = omp_get_max_threads();
    const omp_index globalCount = omp_max_threads * world.n_ranks();

    Aux::Timer diamTimer;
    Aux::Timer phase1Timer;
    Aux::Timer phase1EstimateTimer;
    Aux::Timer phase2Timer;

    // Compute the number of samples per SF as in our EUROPAR'19 paper.
    const auto itersPerStep = std::max(1U,
            static_cast<unsigned int>(baseItersPerStep
                                      / std::pow(globalCount, itersPerStepExp)));

    // TODO: setting the maximum relateve error to 0 gives the exact diameter
    // but may be inefficient for large graphs. What is the maximum relative
    // error that we can tolerate?
    diamTimer.start();
    Diameter diam(G, estimatedRange, .5);
    diam.run();
    // Getting diameter upper bound
    int32_t diameter = diam.getDiameter().second;
    INFO("diameter estimate is [", diam.getDiameter().first, ", ", diameter, ")");
    omega =
        0.5 / err / err * (std::log2(diameter - 1) + 1 + std::log(0.5 / delta));
    diamTime += diamTimer.elapsedMilliseconds();

    const count tau = omega / startFactor;

    if (unionSample == 0) {
        // In the absolute case we need to check that all the estimated
        // betweenness scores are within the error bounds. Thus, we set
        // unionSample to the number of nodes.
        if (absolute) {
            unionSample = n;
        } else {
            unionSample =
                std::min(n, (count)std::max((2 * std::sqrt(G.numberOfEdges()) /
                                             omp_max_threads),
                                            k + 20.));
        }
    }

    if (!absolute)
        this->top = std::unique_ptr<Aux::SortedList>(
            new Aux::SortedList(unionSample, n));

    std::vector<DistributedStateFrame> firstFrames(omp_max_threads, DistributedStateFrame(n));

    phase1Timer.start();
    const int tau_disp = world.rank();
    const int tau_step = world.n_ranks();
#pragma omp parallel for schedule(dynamic)
    for (omp_index i = tau_disp; i < static_cast<omp_index>(tau); i += tau_step) {
        auto t = omp_get_thread_num();
        samplerVec[t].rng.seed(seed0 ^ i);
        samplerVec[t].randomPath(&firstFrames[t]);
    }

    std::vector<count> localBuffer;
    localBuffer.resize(G.upperNodeIdBound() + 1);

    std::vector<count> stagingBuffer;
    stagingBuffer.resize(G.upperNodeIdBound() + 1);

#pragma omp parallel for
    for (omp_index i = 0; i < static_cast<omp_index>(G.upperNodeIdBound()); ++i) {
        for (int j = 0; j < omp_max_threads; ++j)
            localBuffer[i] += firstFrames[j].apx[i];
    }

    if (world.is_rank_zero()) {
        fabry::post(world.reduce(fabry::this_root,
                localBuffer.data(), G.upperNodeIdBound(), stagingBuffer.data()));

        for (count i = 0; i < G.upperNodeIdBound(); ++i)
            approxSum[i] += stagingBuffer[i];
    }else{
        fabry::post(world.reduce(fabry::zero_root,
                localBuffer.data(), G.upperNodeIdBound()));
    }

    nPairs = tau;
    if (!absolute) {
        fillPQ();
    }
    phase1EstimateTimer.start();
    if(world.is_rank_zero())
        computeDeltaGuess();
    phase1EstimateTime += phase1EstimateTimer.elapsedMilliseconds();
    phase1Time += phase1Timer.elapsedMilliseconds();

    epochToRead.store(0, std::memory_order_relaxed);
    nPairs = 0;
    std::fill(approxSum.begin(), approxSum.end(), 0.0);
    epochToRead.store(-1, std::memory_order_relaxed);
    epochRead = -1;
    aggregationDone.resize(omp_max_threads);

    if (!absolute)
        top->clear();

    DistributedStatus status(unionSample);
    phase2Timer.start();
#pragma omp parallel
    {
        const omp_index t = omp_get_thread_num();
        const omp_index globalT = omp_max_threads * world.rank() + t;
        DistributedSpSampler &sampler = samplerVec[t];
        std::deque<DistributedStateFrame *> unused;
        int32_t epochToWrite = 0;
        DistributedStateFrame *curFrame = &firstFrames[t];
        std::deque<DistributedStateFrame *> finishedQueue;
        // Makes sure that dynamically allocated frames will be deallocated
        std::vector<std::unique_ptr<DistributedStateFrame>> additionalFrames;
        maxFrames[t] = 0;

        auto moveToNextEpoch = [&]() {
            ++epochToWrite;
            if (unused.empty()) {
                additionalFrames.push_back(
                    std::unique_ptr<DistributedStateFrame>(new DistributedStateFrame(n)));
                curFrame = additionalFrames.back().get();
                ++maxFrames[t];
            } else {
                curFrame = unused.front();
                unused.pop_front();
            }
            curFrame->reset(epochToWrite);
            sampler.rng.seed(seed1 ^ (epochToWrite * globalCount + globalT));
        };

        auto recycleFrame = [&]() {
            auto finishedFrame =
                epochFinished[t].load(std::memory_order_relaxed);
            if (finishedFrame) {
                unused.push_back(finishedFrame);
            }
        };

        auto doSample = [&]() {
            sampler.randomPath(curFrame);
            curFrame->nPairs++;

            if (deterministic) {
                assert(curFrame->nPairs <= 1000);
                if (curFrame->nPairs == 1000) {
                    finishedQueue.push_back(curFrame);
                    moveToNextEpoch();
                }

                auto etr = epochToRead.load(std::memory_order_relaxed);
                if (!finishedQueue.empty()) {
                    auto frame = finishedQueue.front();
                    auto epochToPublish = static_cast<int32_t>(frame->epoch);
                    assert(etr <= epochToPublish);
                    if (etr == epochToPublish) {
                        recycleFrame();
                        epochFinished[t].store(frame, std::memory_order_release);
                        finishedQueue.pop_front();
                    }
                }
            } else {
                auto etr = epochToRead.load(std::memory_order_relaxed);
                if (etr == epochToWrite) {
                    recycleFrame();
                    epochFinished[t].store(curFrame, std::memory_order_release);
                    moveToNextEpoch();
                }
            }
        };

        curFrame->reset(epochToWrite);
        sampler.rng.seed(seed1 ^ (epochToWrite * globalCount + globalT));

        while (!stop.load(std::memory_order_relaxed)) {
            for (unsigned int i = 0; i < itersPerStep; ++i)
                doSample();

            if (!t) {
                Aux::Timer phase2SyncTimer;
                Aux::Timer phase2TransitionTimer;
                Aux::Timer phase2BarrierTimer;
                Aux::Timer phase2ReduceTimer;
                Aux::Timer phase2CheckTimer;
                Aux::Timer phase2BcastTimer;

                phase2SyncTimer.start();
                // Thread zero also has to check the stopping condition.
                assert(epochToRead.load(std::memory_order_relaxed) == epochRead);

                auto etr = epochRead + 1;
                epochToRead.store(etr, std::memory_order_relaxed);

                // Perform local aggregation.
                phase2TransitionTimer.start();
                std::fill(aggregationDone.begin(), aggregationDone.end(), false);
                {
                    memset(localBuffer.data(), 0, sizeof(count) * (G.upperNodeIdBound() + 1));

                    while(true) {
                        if(aggregateLocally(localBuffer.data()))
                            break;
                        doSample();
                    }
                }
                phase2TransitionTime += phase2TransitionTimer.elapsedMilliseconds();

                phase2BarrierTimer.start();
                fabry::pollable aggBarrier{world.barrier(fabry::collective)};
                while(true) {
                    Aux::StartedTimer ioTimer;
                    bool done = aggBarrier.done();
                    phase2BarrierIoTime += ioTimer.elapsedMilliseconds();
                    if(done)
                        break;
                    Aux::StartedTimer overlapTimer;
                    doSample();
                    phase2BarrierOverlapTime += overlapTimer.elapsedMilliseconds();
                }
                phase2BarrierTime += phase2BarrierTimer.elapsedMilliseconds();

                // Perform RDMA aggregation.
                phase2ReduceTimer.start();
                if(world.is_rank_zero()) {
                    Aux::StartedTimer ioTimer;
                    fabry::post(world.reduce(fabry::this_root,
                            localBuffer.data(), localBuffer.size(), stagingBuffer.data()));
                    phase2ReduceIoTime += ioTimer.elapsedMilliseconds();

                    for (count i = 0; i < G.upperNodeIdBound(); ++i)
                        approxSum[i] += stagingBuffer[i + 1];
                    nPairs += stagingBuffer[0];
                }else{
                    Aux::StartedTimer ioTimer;
                    fabry::post(world.reduce(fabry::zero_root,
                            localBuffer.data(), localBuffer.size()));
                    phase2ReduceIoTime += ioTimer.elapsedMilliseconds();
                }
                phase2ReduceTime += phase2ReduceTimer.elapsedMilliseconds();

                // Check the stopping condition on rank zero.
                phase2BcastTimer.start();
                int globalStop;
                if(world.is_rank_zero()) {
                    phase2CheckTimer.start();
                    globalStop = checkConvergence(status);
                    phase2CheckTime += phase2CheckTimer.elapsedMilliseconds();
                    fabry::pollable convergenceBcast{world.bcast(fabry::this_root, &globalStop)};
                    while(!convergenceBcast.done())
                        doSample();
                }else{
                    fabry::pollable convergenceBcast{world.bcast(fabry::zero_root, &globalStop)};
                    while(!convergenceBcast.done())
                        doSample();
                }
                phase2BcastTime += phase2BcastTimer.elapsedMilliseconds();
                if(globalStop)
                    stop.store(true, std::memory_order_relaxed);

                epochRead++;
                phase2SyncTime += phase2SyncTimer.elapsedMilliseconds();
            }
        }

        // Guarantees that all threads finish the loop here and destroy
        // allocated frames
#pragma omp barrier
    }
    phase2Time += phase2Timer.elapsedMilliseconds();

#pragma omp parallel for
    for (omp_index i = 0; i < static_cast<omp_index>(n); ++i) {
        approxSum[i] /= (double)nPairs;
        if (!G.isDirected())
            approxSum[i] *= 2.;
    }

    if (!absolute) {
        // It should not be necessary to clear it again, but otherwise the
        // ranking is wrong.
        top->clear();
        fillPQ();
    }
    fillResult();
    nPairs += tau;

    hasRun = true;
}

bool DistributedKadabra::aggregateLocally(count *data) {
    bool allEpochsFinished = true;
    const count omp_max_threads = omp_get_max_threads();
    for (count j = 0; j < omp_max_threads; ++j) {
        auto frame = epochFinished[j].load(std::memory_order_acquire);
        if (!frame || static_cast<int32_t>(frame->epoch)
                != epochToRead.load(std::memory_order_relaxed)) {
            allEpochsFinished = false;
            continue;
        }

        // Aggregate data from the thread, but only once.
        if (!aggregationDone[j]) {
            for (count i = 0; i < G.upperNodeIdBound(); ++i)
                data[i + 1] += frame->apx[i];
            data[0] += frame->nPairs;
            aggregationDone[j] = true;
        }
    }

    return allEpochsFinished;
}

bool DistributedKadabra::checkConvergence(DistributedStatus &status) {
/*
    if (!absolute) {
        for (count i = 0; i < n; ++i) {
            top->insert(i, approxSum[i]);
        }
    }
*/

    getStatus(&status);
    return nPairs >= omega || computeFinished(&status);
}

DistributedSpSampler::DistributedSpSampler(const Graph &G, const ConnectedComponents &cc)
    : G(G), cc(cc), rng(0) {
    const auto n = G.upperNodeIdBound();
    distr = std::uniform_int_distribution<node>(0, n - 1);
    q.resize(n);
    timestamp.assign(n, 0);
    dist.assign(n, std::numeric_limits<count>::max());
    nPaths.resize(n);
}

void DistributedSpSampler::randomPath(DistributedStateFrame *curFrame) {
    frame = curFrame;
    node u = distr(rng);
    node v = distr(rng);
    while (u == v)
        v = distr(rng);

    if (DistributedKadabra::determineComponents
            && !G.isDirected() && cc.componentOfNode(u) != cc.componentOfNode(v))
        return;

    count endQ = 2;
    q[0] = u;
    q[1] = v;

    timestamp[u] = globalTS;
    // Setting 8-th bit to 1 (i.e. ball indicator for nodes visited from
    // target).
    timestamp[v] = globalTS + ballMask;

    dist[u] = 0;
    dist[v] = 0;
    nPaths[u] = 1;
    nPaths[v] = 1;

    spEdges.clear();

    node x, randomEdge;
    bool hasToStop = false, useDegreeIn;
    count startU = 0, startV = 1, endU = 1, endV = 2, startCur, endCur,
          *newEndCur;
    count sumDegsU = 0, sumDegsV = 0, *sumDegsCur;
    count totWeight = 0, curEdge = 0;

    auto procNeighbor = [&](const node x, const node y) {
        // Node not visited
        if ((timestamp[y] & stampMask) != globalTS) {
            (*sumDegsCur) += getDegree(G, y, useDegreeIn);
            nPaths[y] = nPaths[x];
            timestamp[y] = globalTS + (timestamp[x] & ballMask);
            q[endQ++] = y;
            ++(*newEndCur);
            dist[y] = dist[x] + 1;
        } else if ((timestamp[x] & ballMask) != (timestamp[y] & ballMask)) {
            hasToStop = true;
            spEdges.push_back(std::make_pair(x, y));
        } else if (dist[y] == dist[x] + 1) {
            nPaths[y] += nPaths[x];
        }
    };

    while (!hasToStop) {
        if (sumDegsU <= sumDegsV) {
            startCur = startU;
            endCur = endU;
            startU = endQ;
            newEndCur = &endU;
            endU = endQ;
            sumDegsU = 0;
            sumDegsCur = &sumDegsU;
            useDegreeIn = false;
        } else {
            startCur = startV;
            endCur = endV;
            startV = endQ;
            newEndCur = &endV;
            endV = endQ;
            sumDegsV = 0;
            sumDegsCur = &sumDegsV;
            useDegreeIn = true;
        }

        while (startCur < endCur) {
            x = q[startCur++];

            if (useDegreeIn)
                G.forInNeighborsOf(x,
                                   [&](const node y) { procNeighbor(x, y); });
            else
                G.forNeighborsOf(x, [&](const node y) { procNeighbor(x, y); });
        }

        if (*sumDegsCur == 0)
            hasToStop = true;
    }

    ++globalTS;

    if (spEdges.size() == 0) {
        resetSampler(endQ);
        if (globalTS == 128) {
            globalTS = 1;
            std::fill(timestamp.begin(), timestamp.end(), 0);
        }
        return;
    }

    for (auto p : spEdges)
        totWeight += nPaths[p.first] * nPaths[p.second];

    std::uniform_int_distribution<node> wDistr(0, totWeight - 1);
    randomEdge = wDistr(rng);

    for (auto p : spEdges) {
        curEdge += nPaths[p.first] * nPaths[p.second];
        if (curEdge > randomEdge) {
            backtrackPath(u, v, p.first);
            backtrackPath(u, v, p.second);
            break;
        }
    }

    if (globalTS == 128) {
        globalTS = 1;
        std::fill(timestamp.begin(), timestamp.end(), 0);
    }

    resetSampler(endQ);
}

void DistributedSpSampler::backtrackPath(const node source, const node target,
                              const node start) {
    if (start == target || start == source)
        return;

    frame->apx[start] += 1;
    count totWeight = nPaths[start];
    std::uniform_int_distribution<node> wDistr(0, totWeight - 1);
    const node randomPred = wDistr(rng);

    node curPred = 0, w = 0;
    bool stop = false;
    // TODO: update this in the case of directed graphs (use inNeighbors if
    // ballind is 0x80)
    G.forNeighborsOf(start, [&](const node t) {
        if (!stop) {
            if (dist[t] == dist[start] - 1 &&
                ((timestamp[start] & ballMask) == (timestamp[t] & ballMask))) {
                w = t;
                curPred += nPaths[target];
                if (curPred > randomPred) {
                    stop = true;
                }
            }
        }
    });

    if (w != source && w != target)
        backtrackPath(source, target, w);
}

void DistributedSpSampler::resetSampler(const count endQ) {
    for (count i = 0; i < endQ; ++i) {
        dist[q[i]] = std::numeric_limits<count>::max();
        nPaths[q[i]] = 0;
    }
}

count DistributedSpSampler::getDegree(const Graph &graph, node z, bool useDegreeIn) {
    return useDegreeIn ? graph.degreeIn(z) : graph.degree(z);
}
} // namespace NetworKit
