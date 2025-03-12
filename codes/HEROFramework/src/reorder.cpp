#include"reorder.hpp"

using namespace std;
// using namespace boost;

bool cmp(const pair<node, int>& x, const pair<node, int>& y) {
    if (x.second == y.second) return x.first < y.first;
    return x.second > y.second;
}

bool cmp2(const pair<node, int>& x, const pair<node, int>& y) {
    if (x.second == y.second) return x.first < y.first;
    return x.second < y.second;
}

void print_new_order(vector<int> &new_order, int left, int right) {
    printf("Current order: ");
    for (int i = left; i < right; ++i) {
        printf("%d ", new_order[i]);
    }
    printf("\n");
}

void HBGP(const Graph &g, vector<int> &new_order, int level) {
    int n = g.get_num_nodes();
    new_order.resize(n);
    int part_size = 2;//PARTITION_SIZE[level];
    int sum_size = 8;//PARTITION_SIZE[level+1];
    int num_part = 4;//64;
    int max_iter = (n / sum_size) + ((n & (sum_size-1)) ? 1 : 0);
    int left = 0, right = 0;
    vector<pair<node, int>> degrees;
    degrees.reserve(sum_size); 
    int *part_cnt = (int*)malloc(sizeof(int) * num_part);
    ull *nbr_labels = (ull*)malloc(sizeof(ull) * n);
    degrees.reserve(sum_size);
    vector<node> neis1;
    vector<node> neis2;
    printf("n: %d, sum_size: %d, num_part: %d, max_iter: %d\n", n, sum_size, num_part, max_iter);
    for (int iter = 0; iter < max_iter; ++iter) {
        // printf("Max iteration: %d, current iteration: %d!\n", max_iter, iter);
        left = sum_size * iter;
        if (iter == max_iter - 1 && (n & (sum_size - 1))) {
            num_part = (n - left) / part_size;
            right = left + num_part * part_size;
        }
        else {
            right = left + sum_size;
        }
        printf("left: %d, right: %d\n", left, right);

        degrees.clear();
        for (int i = left; i < right; ++i) {
            degrees.push_back({i, g.get_number_of_neighbors(i)});
        }
        sort(degrees.begin(), degrees.end(), cmp);
        Double_Linked_List deg(n, n);
        for (auto k : degrees) {
            // printf("add: %ld into deg\n", k.first - left);
            deg.add(k.first - left);
        }
        // print deg
        deg.print();
        printf("key2pos: ");
        for (int i = 0; i < n; ++i) {
            printf("%d ", deg.key2pos[i]);
        }
        printf("\n");

        memset(part_cnt, 0, num_part * sizeof(int));
        memset(nbr_labels, 0, sizeof(ull) * n);

        Linked_List_Heap candidates(num_part * degrees.size());
        int key, v_idx, p_idx;
        printf("degrees.size(): %ld\n", degrees.size());
        candidates.print();
        for (int i = 0; i < degrees.size(); ++i) {
            if (i == 0) {
                key = deg.pop_head();
                v_idx = key / num_part;
                p_idx = key % num_part;
                candidates.reset();
                for (int j = 0; j < num_part; ++j) candidates.del(v_idx * num_part + j);
            }
            else {
                key = candidates.pop();
                v_idx = key / num_part;
                p_idx = key % num_part;
                deg.del(v_idx);
                for (int j = 0; j < num_part; ++j) {
                    if (j == p_idx) continue;
                    candidates.del(v_idx * num_part + j);
                }
            }
            printf("\033[32mkey: %d, v_idx: %d, p_idx: %d\033[0m\n", key, v_idx, p_idx);
            deg.print();
            candidates.print();
            new_order[left + v_idx] = left + p_idx * part_size + part_cnt[p_idx];
            print_new_order(new_order, left, right);
            part_cnt[p_idx]++;
            if (part_cnt[p_idx] == part_size) {
                printf("\033[32mpart_cnt[%d]: %d\033[0m\n", p_idx, part_cnt[p_idx]);
                for (int j = 0; j < right - left; ++j) {
                    if (candidates.in_heap(j * num_part + p_idx))
                        candidates.del(j * num_part + p_idx);
                }
            }

            g.get_neighbors(left + v_idx, neis1);
            if (neis1.size() == 1) continue;
            for (auto u : neis1) {
                if (nbr_labels[u] & ((ull)1 << p_idx)) continue;
                nbr_labels[u] = p_idx;
                if (neis1.size() == 1) continue; 
                g.get_neighbors(u, neis2);
                for (auto w : neis2) {
                    if (w < left || w >= right) continue;
                    printf("v_idx: %d, curr u: %ld, w: %ld\n", v_idx, u, w);
                    w = (w - left) * num_part + p_idx;
                    if (candidates.in_heap(w)) {
                        candidates.inc(w);
                        printf("inc: %ld left: %d, num_part: %d, p_idx: %d\n", w, left, num_part, p_idx);
                    }
                }
            }
            candidates.print();
            printf("\033[32m================\033[0m\n");  
        }
        // print new_order
        print_new_order(new_order, left, right);
    }  
    if (right < n) {
        for (int i = right; i < n; ++i) new_order[i] = i;
    }
    free(nbr_labels); 
    free(part_cnt);
}


void save_vector(string path, const vector<int> &V) {
    ofstream f(path);
    for (auto v : V) {
        f << v << endl;
    }
    f.close();
}

int main() {
    for (int k = 0; k < files.size(); ++k) {
        printf("======================= Graph %s =====================\n", files[k].first.c_str());
        std::cout << "Loading graph " << files[k].first << " from " << dict_path + files[k].second + ".txt" << std::endl;
        Graph g(dict_path + files[k].second + ".txt");
        vector<int> new_id;
        vector<int> final_id;
        vector<int> tmp;
        // for (int i = 2; i >= 0; --i) 
        for (int i = 0; i >= 0; --i) 
        {
            // if (g.get_num_nodes() < PARTITION_SIZE[i]) continue;
            new_id.resize(g.get_num_nodes());
            HBGP(g, new_id, i);
            g.reorder(new_id);
            if (final_id.empty()) final_id = new_id;
            else {
                printf("final_id.size(): %ld, new_id.size(): %ld\n", final_id.size(), new_id.size());
                tmp.resize(new_id.size());
                for (int j = 0; j < new_id.size(); ++j) {
                    tmp[j] = new_id[final_id[j]];
                }
                swap(tmp, final_id);
            }
        }
        g.save(dict_path + "HBGP" + files[k].second + "_HBGPorder.txt");
        save_vector(dict_path + "HBGP" + files[k].second + "_HBGPorder_newID.txt",final_id);
    }

    return 0;
}