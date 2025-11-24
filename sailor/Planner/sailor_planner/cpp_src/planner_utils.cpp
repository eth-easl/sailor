#include "planner_utils.hpp"
#include <cassert>

unordered_map<string, int> GPU_MEMORY = {
    {"A100-40", 40},
    {"V100-16", 16},
    {"GH-96", 96}

};

unordered_map<string, double> GPU_COST = {
    {"A100-40", 4.05},
    {"V100-16", 2.55},
    {"GH-96", 11.06}
};

static void get_combinations(const vector<string> &elements, int k, int start, vector<string> &current_combination, vector<vector<string>> &all_combinations);
static vector<vector<string>> get_permutations(vector<string> elements);

vector<vector<int>> get_stages(int num_layers, int pp)
{
    /**
     * 1) slit the number of layers into PP pipeline parallelism steps
     * 2) residual: extra layers in the last chunk
     * 3) residual = pp - residual i.e. assume 14 layers, PP = 4, chunksize = 3, residual = 2, 4 - 2 = 2 more PP stages
     * 4) Residual essentially will be split evenly among the last residual stages, i.e. if residual is 2, last 2 stages will get chunksize+1 layers
     * 5) stages: a vector with values start, ..., num_layers in stage, where start is the sum(layers) until stage i
     * 6) Stages essentially is a 2D array, where rows correspond to PP stage and columns correspond to the layers in that stage (layer id)
     */

    int chunksize = num_layers / pp;
    int residual = num_layers % pp;
    residual = pp - residual;

    vector<vector<int>> stages = {};
    int start = 0;
    for (int i = 0; i < pp; i++)
    {
        // keep more layers at the last stages to alleviate mem pressure
        int stage_size = i < residual ? chunksize : chunksize + 1;
        vector<int> stage(stage_size);
        iota(begin(stage), end(stage), start);
        start += stage_size;
        stages.push_back(stage);
    }
    return stages;
}

bool check_stage_fits(
    vector<vector<int>> &stages,
    map<pair<int, int>, size_t> &params_per_stage,
    map<pair<int, int>, double> &activation_per_stage,
    int stage_idx,
    int mbs,
    int tmp,
    string gpu_type,
    int float_size)
{
    double gpu_mem_bytes = (size_t)GPU_MEMORY[gpu_type] * 1024 * 1024 * 1024 * 1.0;
    int num_stages = stages.size();
    pair<int, int> key = make_pair(stages[stage_idx][0], stages[stage_idx].size());

    int memory_multiplier_optim = 4 * 2;
    int model_copy = 4;
    int gradients = 4;
    int comm = 4;
    int additional = 4;
    int mf_factor = memory_multiplier_optim + model_copy + gradients + comm + additional;
    double megatron_mem;
    if (tmp == 1)
        megatron_mem = 2.0 * 1e9;
    else
        megatron_mem = 3.0 * 1e9;

    double mf = params_per_stage[key] * (double)mf_factor;
    double af_stage = activation_per_stage[key] * mbs * float_size;

    // printf(
    //     "Stage %d, Num layers %d, params %lu, mf %f GB, af is %f floats, af_stage is %f GB, total is %f\n",
    //     stage_idx, stages[stage_idx].size(), params_per_stage[key], mf/(1e9),
    //     activation_per_stage[key] * mbs, af_stage/1e9, (mf+af_stage*(num_stages - stage_idx)+megatron_mem)/1e9
    // );

    double mem_used = mf + af_stage * (num_stages - stage_idx) + megatron_mem;

    return (mem_used <= gpu_mem_bytes);
}

void print_vector(vector<double> v)
{
    cout << "[";
    for (auto el : v)
    {
        cout << el << ",";
    }
    cout << "]";
    cout << endl;
}

void print_vector_int(vector<int> v)
{
    cout << "[";
    for (auto el : v)
    {
        cout << el << ",";
    }
    cout << "]";
    cout << endl;
}

pair<double, double> find_p2p_time_cost(
    int stage_idx,
    int num_stages,
    int config_idx,
    int num_configs,
    double activation_size,
    vector<struct StageConfig> &configs_per_stage,
    NETWORK_COEFFS_TYPE &network_coeff,
    unordered_map<size_t, double> &known_times,
    unordered_map<string, unordered_map<string, double>> &comm_cost,
    vector<string> &id_to_zone,
    bool activation)
{
    // compute time needed to send (and receive) activations

    if (activation && (stage_idx == num_stages - 1))
        return make_pair(0.0, 0.0);

    if (!activation && (stage_idx == 0))
        return make_pair(0.0, 0.0);

    // // 1. find sender-receiver
    auto sender = configs_per_stage[stage_idx].dp_pairs[config_idx];
    // TODO: fix the config index - fix when deciding on DP!
    int p2p_idx = activation ? stage_idx + 1 : stage_idx - 1;

    //printf("DP1: %d, DP2: %d\n", configs_per_stage[stage_idx].dp_pairs.size(), configs_per_stage[p2p_idx].dp_pairs.size());

    auto receiver = configs_per_stage[p2p_idx].dp_pairs[config_idx];
    string sender_zone = id_to_zone[get<2>(sender)];
    int sender_gpu_idx = get<0>(sender);
    int sender_gpu_cnt = get<1>(sender);

    string recver_zone = id_to_zone[get<2>(receiver)];
    int recver_gpu_idx = get<0>(receiver);
    int recver_gpu_cnt = get<1>(receiver);

    auto network_coefficients = network_coeff[sender_zone][sender_gpu_idx][sender_gpu_cnt][recver_zone][recver_gpu_idx][recver_gpu_cnt].first;
    //auto network_coefficients = network_coeff[sender_zone][sender_gpu_idx][sender_gpu_cnt][recver_zone][sender_gpu_idx][sender_gpu_cnt].first;

    // cout << sender_zone << "," << sender_gpu_idx << "," << sender_gpu_cnt << endl;
    // cout << recver_zone << "," << recver_gpu_idx << "," << recver_gpu_cnt << endl;

    // printf("Before estimate send time and cost\n");

    double p2p_time = estimate_send_time(activation_size, network_coefficients, known_times);
    double p2p_cost = comm_cost[sender_zone][recver_zone] * activation_size / 1e9;

    auto res = make_pair(p2p_time, p2p_cost);
    return res;
}

double find_ar_cost(
    double tensor_size,
    int dp,
    set<string> &zones
)
{

    if (dp == 1)
        return 0.0;

    double ar_cost = 0.0;
    double zone_cost_gb = 0.01;

    // get zones
    int num_zones = zones.size();

    // we make the following simplifications, based on the algorithm:
    // 1. we know we don't do DP across regions, so any costs will arise from cross-zone communication
    // 2. we know that across-zone communication cost is the same in GCP (TODO: check for others)
    // 3. the algorithm assigns dp degress "sequentially", i.e. in an all-reduce ring, if N zones are there, there will be N cutpoints
    if (num_zones > 1)
    {
        double tensor_size_per_transfer = tensor_size / dp;
        ar_cost = 2 * (dp - 1) * (num_zones * tensor_size_per_transfer * zone_cost_gb) / 1e9;
    }

    return ar_cost;
}

double find_ar_time(
    double tensor_size,
    NETWORK_COEFFS_TYPE &network_coeff,
    vector<pair<int, int>> &tp_degrees,
    set<string> &zones,
    vector<string>& id_to_zone,
    map<pair<string, string>, vector<vector<vector<map<pair<int, int>, double>>>>> &ar_times_bottleneck,
    pair<int, size_t> stage_key,
    int dp)
{
    if (dp == 1)
        return 0.0;

    int gpu_idx;
    int tp = 1;
    int min_tmp = 4;

    // printf("Inside find_ar_time, dp size is %d\n", dp);
    double ar_part = tensor_size / dp;
    double min_network_bw = 1000.0 * 1e9;

    string min_szone, min_rzone;

    for (int i = 0; i < tp_degrees.size(); i++)
    {
        int tpi = tp_degrees[i].first;
        if (tpi == 0)
            continue;
        min_tmp = min(min_tmp, tpi);
        string sender_zone = id_to_zone[tp_degrees[i].second];

        for (auto receiver_zone = zones.begin(); receiver_zone != zones.end(); receiver_zone++)
        {
            //auto network_coeff_config = network_coeff[sender_zone][i][tpi][*receiver_zone][i][tpi].first;
            double bw_bytes = network_coeff[sender_zone][i][tpi][*receiver_zone][i][tpi].second; //get_network_bandwidth(ar_part, network_coeff_config);
            if (bw_bytes < min_network_bw)
            {
                min_network_bw = min(min_network_bw, bw_bytes);
                gpu_idx = i;
                tp = tpi;
                min_szone = sender_zone;
                min_rzone = sender_zone;
            }
        }
    }


    auto x =  ar_times_bottleneck[make_pair(min_szone, min_rzone)][gpu_idx][min_tmp][dp][stage_key];
    //cout << "x is: " << x << endl;
    return x;
}

pair<double, double> merge_stages_get_time(
    struct PipelineInfo *pp_info1,
    struct PipelineInfo *pp_info2,
    struct TrainingInfo *training_info,
    int min_dp,
    int mbs,
    int float_size,
    int stage_idx,
    int num_stages,
    vector<vector<int>> &stages,
    vector<struct StageConfig> &configs_per_stage,
    NETWORK_COEFFS_TYPE &network_coeff,
    unordered_map<size_t, double> &known_times,
    unordered_map<string, unordered_map<string, double>> &comm_cost,
    vector<string>& id_to_zone
)
{
    // Returns a pair of <throughput, pipeline comm cost>
    auto stage = stages[stage_idx];
    int num_micro_batches = ceil(training_info->global_batch_size * 1.0 / (mbs * min_dp));
    double out_stage_act = (training_info->model).out_params[1][stage.back()] * float_size * mbs;
    double stage_time_comm_act = 0.0;
    double stage_time_comm_grad = 0.0;
    double pp_cost = 0.0;

    auto config = configs_per_stage[0]; // for the current stage

    for (int k = 0; k < config.dp; k++)
    {
        pair<double, double> res = find_p2p_time_cost(
            stage_idx,
            num_stages,
            k,
            config.dp,
            out_stage_act,
            configs_per_stage,
            network_coeff,
            known_times,
            comm_cost,
            id_to_zone,
            true);
        double stage_time_comm_pair = res.first;
        pp_cost += res.second * num_micro_batches;
        stage_time_comm_act = max(stage_time_comm_act, stage_time_comm_pair);
    }

    stage_time_comm_grad = stage_time_comm_act;
    pp_cost *= 2;
    // for (int k = 0; k < config.dp; k++)
    // {
    //     pair<double, double> res = find_p2p_time_cost(
    //         stage_idx + 1,
    //         num_stages,
    //         k,
    //         config.dp,
    //         out_stage_act,
    //         configs_per_stage,
    //         network_coeff,
    //         known_times,
    //         comm_cost,
    //         id_to_zone,
    //         false);
    //     double stage_time_comm_pair = res.first;
    //     pp_cost += res.second;
    //     stage_time_comm_grad = max(stage_time_comm_grad, stage_time_comm_pair);
    // }

    // printf("************************************************************ STAGE IDX IS %d\n", stage_idx);

    double straggler = max(pp_info1->straggler + stage_time_comm_act, pp_info2->first_stage_comp + stage_time_comm_grad);
    straggler = max(straggler, pp_info2->straggler);
    double straggler_overhead = (num_micro_batches - 1) * straggler;
    double tot_communication_time = pp_info1->inter_stage_comm + pp_info2->inter_stage_comm + stage_time_comm_act + stage_time_comm_grad;

    // printf("******* Stage comm is %f, %f\n", stage_time_comm_act, stage_time_comm_grad);
    // printf("******* Straggler: %f\n", straggler);
    // printf("******* Straggler Overhead: %f\n", straggler_overhead);
    // printf("******* Communication time: %f\n", tot_communication_time);

    // printf("AR_SYNC1 is %f, AR_SYNC2 is %f\n", pp_info1->ar_sync, pp_info2->ar_sync);

    double tot_computation_time = pp_info1->comp_time + pp_info2->comp_time;
    double tsync = max(pp_info1->ar_sync, pp_info2->ar_sync);
    double update_time = 0.0; // max(pp_info1->update, pp_info2->update);

    // printf("******* Computation time: %f\n", tot_computation_time);
    // printf("******* Update time: %f\n", update_time);
    // printf("******* Sync time: %f\n", tsync);
    // printf("************************************************************ STAGE IDX IS %d\n", stage_idx);

    double Tpp = straggler_overhead + tot_communication_time + tot_computation_time + update_time + tsync;

    pp_info1->straggler = straggler;
    pp_info1->first_stage_comp = pp_info1->comp_time + stage_time_comm_act;
    pp_info1->comp_time = tot_computation_time;
    pp_info1->inter_stage_comm = tot_communication_time;
    pp_info1->ar_sync = tsync;
    pp_info1->Tpp = Tpp;
    pp_info1->update = update_time;

    auto Tpp_cost = make_pair(Tpp, pp_cost);
    return Tpp_cost;
}

struct PipelineInfo simulate_time_single_stage(
    vector<int> &stage,
    vector<pair<int, int>> &tp_degrees,
    set<string> &zones,
    vector<string>& id_to_zone,
    struct TrainingInfo *training_info,
    int mbs,
    NETWORK_COEFFS_TYPE &network_coeff,
    map<pair<string, string>, vector<vector<vector<map<pair<int, int>, double>>>>> &ar_times_bottleneck,
    double stage_params_size,
    int dp
)
{

    double stage_time_comp = 0.0;
    double update_time = 0.0;

    // find time for this stage - get the unique ones only!
    for (int gpu_idx = 0; gpu_idx < tp_degrees.size(); gpu_idx++)
    {
        if (tp_degrees[gpu_idx].first == 0)
            continue;

        double stage_time_comp_pair = 0.0;
        int tp = tp_degrees[gpu_idx].first;
        for (auto layer : stage)
        {
            stage_time_comp_pair += (training_info->model).profiles_per_gpu[gpu_idx][make_pair(mbs, tp)].exec_times[layer];
        }
        stage_time_comp = max(stage_time_comp, stage_time_comp_pair);
        update_time = max(update_time, stage.size() * (training_info->model).profiles_per_gpu[gpu_idx][make_pair(mbs, tp)].update);
    }

    // printf("COMP TIME PER STAGE IS %f\n", stage_time_comp);

    int num_micro_batches = ceil(training_info->global_batch_size * 1.0 / (mbs * dp));
    double straggler = stage_time_comp;
    double all_comp_time = num_micro_batches * stage_time_comp;

    double inter_stage_comm = 0;
    auto stage_key = make_pair(stage[0], stage.size());
    double ar_sync = find_ar_time(stage_params_size, network_coeff, tp_degrees, zones, id_to_zone, ar_times_bottleneck, stage_key, dp);
    double ar_cost = find_ar_cost(stage_params_size, dp, zones);

    //printf("ALL_COMP_TIME: %f, AR_SYNC: %f, UPDATE_TIME: %f\n", all_comp_time, ar_sync, update_time);

    double Tpp = all_comp_time + ar_sync + update_time;

    return PipelineInfo(straggler, straggler, inter_stage_comm, ar_sync, update_time, Tpp, stage_time_comp, ar_cost);
}

double get_cost_gpu_type(int num_gpus_used, string gpu_type)
{
    return num_gpus_used * GPU_COST[gpu_type];
}

double get_ar_time_with_buckets(
    size_t tensor_size,
    double bucket_size,
    int num_workers,
    vector<double> &network_coeff,
    unordered_map<size_t, double> known_times)
{
    size_t remaining = tensor_size;
    double dp_time = 0.0;
    // printf("%lu, %f, %d\n", tensor_size, bucket_size, num_workers);
    // for (auto cf: network_coeff)
    //     cout << cf << endl;
    while (remaining > 0)
    {
        size_t send_size = remaining > bucket_size ? bucket_size : remaining;
        dp_time += estimate_ar_time(
            send_size,
            network_coeff,
            num_workers,
            known_times);
        remaining -= send_size;
    }
    return dp_time;
}

vector<vector<int>> find_tmp_degrees(
    vector<vector<int>> &stages,
    struct TrainingInfo *training_info,
    vector<vector<map<pair<int, int>, int>>> &max_tmps_vector_per_gpu,
    vector<vector<map<tuple<int, int, int>, int>>> min_tmps_vector_per_gpu,
    vector<vector<int>> &possible_tmps,
    int mbs,
    int num_available_gpus,
    int float_size,
    bool homog
)
{
    int num_stages = stages.size();
    vector<vector<int>> min_tmps;
    vector<vector<int>> tmps;

    // 1. find min TMPs

    for (int gpu_idx = 0; gpu_idx < num_available_gpus; gpu_idx++)
    {
        min_tmps.push_back({});
        bool all_tp_found = true;
        for (int i = 0; i < num_stages; i++)
        {
            auto stage = stages[i];
            tuple<int, int, int> key(num_stages, stage[0], stage.size());
            if (min_tmps_vector_per_gpu[gpu_idx].size() < mbs) {
                all_tp_found = false;
                break;
            }

            if (mbs >= min_tmps_vector_per_gpu[gpu_idx].size()) {
                all_tp_found = false;
                break;
            }
            int tp_stage = min_tmps_vector_per_gpu[gpu_idx][mbs][key];
            if (tp_stage == -1)
            {
                all_tp_found = false;
                break;
            }
            else
            {
                min_tmps[gpu_idx].push_back(tp_stage);
            }
        }
        if (!all_tp_found)
        {
            min_tmps[gpu_idx] = {};
        }
    }

    for (int gpu_idx = 0; gpu_idx < num_available_gpus; gpu_idx++)
    {
        cout << "MIN_TMPS - Check for GPU " << gpu_idx << endl;
        print_vector_int(min_tmps[gpu_idx]);
    }

    // 2. find speedup-based TMPs
    for (int gpu_idx = 0; gpu_idx < num_available_gpus; gpu_idx++)
    {
        tmps.push_back({});
        if (min_tmps[gpu_idx].empty())
        {
            tmps[gpu_idx] = {};
            continue;
        }
        tmps[gpu_idx] = {};
        for (int i = 0; i < num_stages; i++)
        {
            auto stage = stages[i];
            auto key = make_pair(stage[0], stage.size());
            if (gpu_idx==0) {
                if (homog)
                    tmps[gpu_idx].push_back(max(max_tmps_vector_per_gpu[gpu_idx][mbs][key], min_tmps[gpu_idx][i]));
                else
                    tmps[gpu_idx].push_back(max(1, min_tmps[gpu_idx][i]));
            } else
                tmps[gpu_idx].push_back(max(max_tmps_vector_per_gpu[gpu_idx][mbs][key], min_tmps[gpu_idx][i]));
        }
    }


    return tmps;
}

string extract_region_from_zone(const string &zone)
{
    /**
     * Given a zone of format "<continent>-<region>-<zone>" (e.g. "us-central1-a"),
     * extract and return the region information (e.g. "us-central1")
     * @param zone: zone string
     * @return region: extracted region from the input zone string
     */
    // Find the first delimiter
    size_t firstDash = zone.find('-');
    if (firstDash == string::npos)
    {
        // If no delimiter is found, return the original string
        return zone;
    }

    // Find the second delimiter
    size_t secondDash = zone.find('-', firstDash + 1);
    if (secondDash == string::npos)
    {
        // If no second delimiter is found, return the string up to the first delimiter
        return zone;
    }

    // Extract the substring up to the second delimiter
    return zone.substr(0, secondDash);
}

double get_max_throughput(const NETWORK_COEFFS_TYPE &network_coeff,
                                 const string &zone1, int gpu_count1,
                                 const string &zone2, int gpu_count2)
{
    /**
     * Helper function that finds the maximum network TP among 2 zones for given gpu counts
     * @param network_coeff: network_coefficients that include the maximum TP among 2 zones aside from the coefficients
     * @param zone1, zone2: selected zones to lookup maximum TP
     * @param gpu_count1, gpu_count2: gpu_counts for zone1 and 2 respectively
     */
    double max_throughput = -1.0;
    auto it_sender_zone = network_coeff.find(zone1);
    if (it_sender_zone != network_coeff.end())
    {
        const auto &sender_gpu_types = it_sender_zone->second;
        for (const auto &sender_gpu_type_entry : sender_gpu_types)
        {
            const auto &sender_gpu_counts = sender_gpu_type_entry.second;

            // Directly look for the specific sender GPU count
            auto it_sender_gpu_count = sender_gpu_counts.find(gpu_count1);
            if (it_sender_gpu_count != sender_gpu_counts.end())
            {
                const auto &receiver_zones = it_sender_gpu_count->second;

                auto it_receiver_zone = receiver_zones.find(zone2);
                if (it_receiver_zone != receiver_zones.end())
                {
                    const auto &receiver_gpu_types = it_receiver_zone->second;
                    for (const auto &receiver_gpu_type_entry : receiver_gpu_types)
                    {
                        const auto &receiver_gpu_counts = receiver_gpu_type_entry.second;

                        // Directly look for the specific receiver GPU count
                        auto it_receiver_gpu_count = receiver_gpu_counts.find(gpu_count2);
                        if (it_receiver_gpu_count != receiver_gpu_counts.end())
                        {
                            const auto &data = it_receiver_gpu_count->second;
                            double curr_max_throughput = data.second / 1000000.0; // Second element is max throughput

                            if (curr_max_throughput > max_throughput)
                            {
                                max_throughput = curr_max_throughput;
                            }
                        }
                    }
                }
            }
        }
    }
    return max_throughput;
}

static void get_combinations(const vector<string> &elements, int k, int start, vector<string> &current_combination, vector<vector<string>> &all_combinations)
{
    /**
     * Helper function that generates all possible combinations of the elements array using n-select-k
     * @param elements: available regions list
     * @param k: k elements to select
     * @param start: starting index (should be 0 initially, used for recursion)
     * @param current_combination: auxiliary vector, should be empty
     * @param all_combinations: actual return vector
     */
    if (k == 0)
    {
        all_combinations.push_back(current_combination);
        return;
    }
    for (int i = start; i <= elements.size() - k; ++i)
    {
        current_combination.push_back(elements[i]);
        get_combinations(elements, k - 1, i + 1, current_combination, all_combinations);
        current_combination.pop_back();
    }
}

static vector<vector<string>> get_permutations(vector<string> elements)
{
    /**
     * Given a sequence, get all possible permutations (i.e. input: 1-2-3, output: 1-2-3, 3-2-1, 2-1-3, 2-3-1, 3-1-2)
     * @param elements: sequence of strings
     * @return permutations: all possible permutations of the string sequence
     */
    vector<vector<string>> permutations;
    sort(elements.begin(), elements.end());
    do
    {
        permutations.push_back(elements);
    } while (next_permutation(elements.begin(), elements.end()));
    return permutations;
}

vector<pair<string, vector<string>>> get_regions_list(
    const unordered_map<string,vector<pair<string, vector<int>>>> &given_resources,
    unordered_map<string, vector<int>> &region_gpu_count,
    unordered_map<string, vector<string>> &zones_per_region,
    vector<string> regions,
    unordered_map<string, unordered_map<string, double>> &throughput,
    const NETWORK_COEFFS_TYPE &network_coeff,
    const vector<vector<int>> &tmp_degrees
)
{
    /**
     *   Given some resources scattered in multiple regions, return a list of topologically ordered region lists that can support the pipeline
     *   @param given_resources: map of type {"region" : [("zone", gpus)]}
     *   @param network_coeffs: network coefficients
     *   @param tmp_degrees: tensor parallelism per gpu_type and per stage of PP parallelism
     *   @return regions: vector of (hash_string, region_list_i) elements, where region_list_i has i elements for i = 0, ..., num_regions
     */
    // Number of regions
    int N = (int)given_resources.size();

    // Extract regions and compute total GPUs per type
    //vector<string> regions;
    // Assume GPU indexing in given_resources[region][zone_idx] matches tmp_degrees indexing
    int gpu_types = (int)tmp_degrees.size();

    // Compute required GPUs per type from tmp_degrees
    // A valid region sequence should have at least as many resources of any gpu type
    vector<int> required_per_type(gpu_types, 0);
    for (int i = 0; i < gpu_types; ++i)
    {
        if (!tmp_degrees[i].empty())
            required_per_type[i] = accumulate(tmp_degrees[i].begin(), tmp_degrees[i].end(), 0);
    }


    // Now, consider all subsets of regions (from size=1 to N) and their permutations
    // We store all feasible permutations along with their throughput
    struct Candidate
    {
        double total_throughput;
        vector<string> ordering;
        string hash_string;
    };
    vector<Candidate> feasible_candidates;

    for (int subset_size = 1; subset_size <= N; ++subset_size)
    {
        vector<vector<string>> combinations;
        vector<string> current_combination;

        // TODO{OPTIM}: move outside?
        get_combinations(regions, subset_size, 0, current_combination, combinations);

        for (const auto &comb : combinations)
        {
            // Check if this combination can support the pipeline
            // Sum GPUs of each type
            vector<int> sum_gpus(gpu_types, 0);
            string hash_string = "";
            for (const auto &r : comb)
            {
                const vector<int> &gcounts = region_gpu_count[r];
                hash_string += "_" + r;
                for (int i = 0; i < gpu_types; ++i)
                {
                    hash_string += "_" + to_string(i) + "-" + to_string(gcounts[i]);
                    sum_gpus[i] += gcounts[i];
                }
            }

            // Check feasibility: at least one type i satisfies sum_gpus[i] >= required_per_type[i]
            bool feasible = false;
            for (int i = 0; i < gpu_types; ++i)
            {
                if (sum_gpus[i] >= required_per_type[i])
                {
                    feasible = true;
                    break;
                }
            }
            if (!feasible)
                continue;

            vector<vector<string>> perms = get_permutations(comb);
            double max_th = -1;
            vector<string> max_perm;
            for (const auto &perm : perms)
            {
                double total_th = 10000.0; // random large number to prioritize 1 region options
                if (subset_size > 1)
                { // No need to compute TP for 1 region
                    // Calculate bottleneck throughput
                    total_th = throughput[perm[0]][perm[1]];
                    for (size_t i = 1; i < perm.size() - 1; ++i)
                    {
                        const string &r1 = perm[i];
                        const string &r2 = perm[i + 1];
                        total_th = min(throughput[r1][r2], total_th);
                    }
                    if (total_th > max_th)
                    { // find permutation with max throughput
                        max_th = total_th;
                        max_perm = perm;
                    }
                }
                else
                {
                    max_th = total_th;
                    max_perm = perm;
                }
            }
            feasible_candidates.push_back({max_th, max_perm, hash_string});
        }
    }

    // Sort feasible candidates by total_throughput in descending order
    sort(feasible_candidates.begin(), feasible_candidates.end(),
         [](const Candidate &a, const Candidate &b)
         {
             return a.total_throughput > b.total_throughput;
         });

    // Extract just the orderings and the hash strings
    vector<pair<string, vector<string>>> selected_regions;
    for (auto &cand : feasible_candidates)
    {
        selected_regions.push_back(make_pair(cand.hash_string, cand.ordering));
    }

    return selected_regions;
}