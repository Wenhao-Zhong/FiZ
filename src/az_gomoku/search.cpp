// Independent AlphaZero-style PUCT. Full legal move set; no tactical pruning,
// opening book, solver, handcrafted evaluation or external teacher.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <array>
#include <vector>
#include <memory>
#include <random>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <limits>

namespace py = pybind11;
constexpr int S = 15, A = S*S;
struct Position {
    std::array<int8_t,A> board{};
    int color=1, last=-1, ply=0, winner=0;
    bool terminal=false;
    Position move(int action) const {
        if (terminal || action<0 || action>=A || board[action]) throw std::runtime_error("Illegal move");
        Position next=*this;
        next.board[action]=color; next.last=action; next.ply++; next.color=-color;
        int row=action/S, col=action%S;
        for (auto d : {std::pair<int,int>{0,1},{1,0},{1,1},{1,-1}}) {
            int count=1;
            for (int sign : {-1,1}) {
                int r=row+sign*d.first, c=col+sign*d.second;
                while(r>=0 && r<S && c>=0 && c<S && next.board[r*S+c]==color) {
                    count++; r+=sign*d.first; c+=sign*d.second;
                }
            }
            if(count>=5) {next.terminal=true; next.winner=color;}
        }
        next.terminal=next.terminal || next.ply==A;
        return next;
    }
};
struct Node {
    Position pos;
    int visits=0, virtuals=0;
    double total=0, network_value=0;
    int forced_playouts=0;
    bool expanded=false, pending=false;
    std::array<float,A> prior{}, base_prior{};
    std::array<std::unique_ptr<Node>,A> children;
    explicit Node(const Position& p):pos(p){}
};
struct Pending {int tree; std::vector<Node*> path;};

class Forest {
    std::vector<std::unique_ptr<Node>> roots;
    std::vector<int> completed, targets, max_depth;
    std::vector<long long> total_depth;
    std::vector<bool> exploratory;
    std::unordered_map<int,Pending> pending;
    std::mt19937_64 rng;
    int serial=0;
    double cpuct=1.5, epsilon=0, alpha=.03;
    double fpu_reduction=.2, root_fpu_reduction=.2, forced_coeff=0;
    long long actual_evaluations=0, simulations=0;
    void noise(Node& root,int tree) {
        root.prior=root.base_prior;
        if(epsilon<=0 || !exploratory[tree]) return;
        std::gamma_distribution<double> distribution(alpha,1.);
        std::array<double,A> values{};
        double sum=0;
        for(int a=0;a<A;a++) if(!root.pos.board[a]) { values[a]=distribution(rng); sum+=values[a]; }
        if(sum>0) for(int a=0;a<A;a++) if(!root.pos.board[a])
            root.prior[a]=(1-epsilon)*root.base_prior[a]+epsilon*values[a]/sum;
    }
    static void backup(std::vector<Node*>& path,double value,bool release) {
        // value is from the leaf side-to-move; every edge changes perspective.
        for(auto it=path.rbegin();it!=path.rend();++it) {
            Node* node=*it;
            if(release) {if(node->virtuals<=0) throw std::runtime_error("Virtual loss underflow"); node->virtuals--;}
            node->visits++; node->total+=value; value=-value;
        }
    }
    // Both the parent network estimate and the accumulated search estimate are
    // in this node's side-to-move perspective. Unseen moves inherit that scale,
    // rather than acquiring an artificial draw value in a nearly lost game.
    double fpu(const Node& node,bool at_root,int tree) const {
        double explored_policy=0;
        for(int a=0;a<A;a++) if(node.children[a] &&
             node.children[a]->visits+node.children[a]->virtuals>0)
            explored_policy+=node.prior[a];
        explored_policy=std::min(1.,std::max(0.,explored_policy));
        double searched=node.visits>0 ? node.total/node.visits : node.network_value;
        double parent=(1.-explored_policy)*node.network_value+explored_policy*searched;
        double reduction=at_root ? root_fpu_reduction : fpu_reduction;
        if(at_root && exploratory[tree] && epsilon>0) reduction=0.;
        return parent-reduction*std::sqrt(explored_policy);
    }
    std::vector<Node*> select(Node* node,int tree,bool& forced) {
        std::vector<Node*> path{node};
        forced=false;
        while(node->expanded && !node->pos.terminal) {
            double best=-1e300;
            int action=-1, ties=0;
            bool at_root=path.size()==1, selected_forced=false;
            double scale=cpuct*std::sqrt(std::max(1,node->visits+node->virtuals));
            double unseen=fpu(*node,at_root,tree);
            for(int a=0;a<A;a++) {
                if(node->pos.board[a]) continue;
                Node* child=node->children[a].get();
                int count=child ? child->visits+child->virtuals : 0;
                double q=count ? -(child->total+child->virtuals)/count : unseen;
                double score=q+scale*node->prior[a]/(1+count);
                bool force=at_root && exploratory[tree] && forced_coeff>0 && count>0 &&
                    count<std::sqrt(forced_coeff*node->prior[a]*std::max(1,node->visits+node->virtuals));
                if(force) score=std::numeric_limits<double>::infinity();
                if(score>best) {best=score;action=a;ties=1;selected_forced=force;}
                else if(score==best) {
                    // Reservoir sampling is uniform over tied candidates even
                    // when their coordinates are adjacent or unevenly spaced.
                    ++ties;
                    if(std::uniform_int_distribution<int>(1,ties)(rng)==1) {
                        action=a;selected_forced=force;
                    }
                }
            }
            if(action<0) throw std::runtime_error("No legal action at nonterminal node");
            if(at_root) forced=selected_forced;
            if(!node->children[action]) node->children[action]=std::make_unique<Node>(node->pos.move(action));
            node=node->children[action].get(); path.push_back(node);
        }
        return path;
    }
    std::array<double,A> visit_counts(int tree,bool prune_forced) const {
        const Node& root=*roots[tree];
        std::array<double,A> counts{};
        int maxn=0; double total=0;
        for(int a=0;a<A;a++) if(root.children[a]) {
            counts[a]=root.children[a]->visits;
            maxn=std::max(maxn,root.children[a]->visits); total+=counts[a];
        }
        if(!prune_forced || !exploratory[tree] || forced_coeff<=0 || maxn==0) return counts;
        double scale=cpuct*std::sqrt(std::max(1.,total));
        double best_score=-std::numeric_limits<double>::infinity();
        for(int a=0;a<A;a++) if(counts[a]==maxn) {
            const Node& child=*root.children[a];
            best_score=std::max(best_score,-child.total/child.visits+scale*root.prior[a]/(1+counts[a]));
        }
        // KataGo-style *target* pruning: remove at most the forced exploration
        // allowance while holding Q fixed. This never changes the search tree,
        // action selection, or legal candidates. Preserve every maximum-count
        // child, including the pathological all-tied case, so targets stay valid.
        for(int a=0;a<A;a++) if(counts[a]>0 && counts[a]<maxn) {
            const Node& child=*root.children[a];
            double q=-child.total/child.visits;
            int allowance=static_cast<int>(std::sqrt(forced_coeff*root.prior[a]*total));
            for(int removed=0;removed<allowance && counts[a]>0;removed++) {
                double proposed=counts[a]-1;
                if(q+scale*root.prior[a]/(1+proposed)>=best_score) break;
                counts[a]=proposed;
            }
            // Preserve a new high-Q singleton unless pruning actually reduced it.
            if(counts[a]<=1 && counts[a]<child.visits) counts[a]=0;
        }
        return counts;
    }
public:
    Forest(int batch,unsigned long long seed):rng(seed) {
        if(batch<=0) throw std::runtime_error("Batch must be positive");
        for(int i=0;i<batch;i++) roots.push_back(std::make_unique<Node>(Position{}));
        completed.resize(batch); targets.resize(batch); exploratory.resize(batch);
        max_depth.resize(batch);total_depth.resize(batch);
    }
    void begin(int sims,double c,double e,double al,double reduction=.2,double root_reduction=.2,
               double forced=0.,const std::vector<int>& budgets={},const std::vector<bool>& exploration={}) {
        if(!pending.empty() || sims<1 || !std::isfinite(c) || c<=0 || !std::isfinite(e) ||
           e<0 || e>1 || !std::isfinite(al) || al<=0 || !std::isfinite(reduction) || reduction<0 ||
           !std::isfinite(root_reduction) || root_reduction<0 || !std::isfinite(forced) || forced<0 ||
           (!budgets.empty() && budgets.size()!=roots.size()) ||
           (!exploration.empty() && exploration.size()!=roots.size()))
            throw std::runtime_error("Invalid search settings/state");
        for(int budget:budgets) if(budget<1) throw std::runtime_error("Visit caps must be positive");
        cpuct=c; epsilon=e; alpha=al;fpu_reduction=reduction;root_fpu_reduction=root_reduction;forced_coeff=forced;
        std::fill(completed.begin(),completed.end(),0);
        std::fill(max_depth.begin(),max_depth.end(),0);
        std::fill(total_depth.begin(),total_depth.end(),0);
        for(int i=0;i<(int)roots.size();i++) {
            exploratory[i]=exploration.empty() ? e>0 : exploration[i];
            // A budget is a total visit cap, not an additional-playout request.
            // Retained actor trees therefore save work on subsequent moves.
            targets[i]=roots[i]->pos.terminal ? 0 : std::max(0,(budgets.empty()?sims:budgets[i])-roots[i]->visits);
            if(roots[i]->expanded && !roots[i]->pos.terminal) noise(*roots[i],i);
        }
    }
    py::tuple request(int per_tree) {
        if(!pending.empty() || per_tree<1) throw std::runtime_error("Outstanding requests or invalid batch");
        std::vector<int> ids,colors,last;
        std::vector<Position> positions;
        for(int i=0;i<(int)roots.size();i++) {
            if(roots[i]->pos.terminal) { completed[i]=targets[i]; continue; }
            int outstanding=0;
            while(outstanding<per_tree && completed[i]+outstanding<targets[i]) {
                bool forced=false;
                auto path=select(roots[i].get(),i,forced); Node* leaf=path.back();
                if(leaf->pos.terminal) {
                    double value=leaf->pos.winner ? (leaf->pos.winner==leaf->pos.color ? 1.:-1.) : 0.;
                    max_depth[i]=std::max(max_depth[i],(int)path.size()-1);total_depth[i]+=path.size()-1;
                    roots[i]->forced_playouts+=forced;
                    backup(path,value,false); completed[i]++; simulations++; continue;
                }
                if(leaf->pending) break;
                max_depth[i]=std::max(max_depth[i],(int)path.size()-1);total_depth[i]+=path.size()-1;
                roots[i]->forced_playouts+=forced;
                leaf->pending=true;
                for(Node* n:path) n->virtuals++;
                int id=serial++;
                pending.emplace(id,Pending{i,std::move(path)});
                ids.push_back(id); positions.push_back(leaf->pos); colors.push_back(leaf->pos.color); last.push_back(leaf->pos.last);
                outstanding++;
            }
        }
        py::array_t<int8_t> boards({(int)ids.size(),A});
        auto b=boards.mutable_unchecked<2>();
        for(int i=0;i<(int)positions.size();i++) for(int a=0;a<A;a++) b(i,a)=positions[i].board[a];
        return py::make_tuple(ids,boards,colors,last);
    }
    void complete(const std::vector<int>& ids,
                  py::array_t<float,py::array::c_style|py::array::forcecast> probs,
                  py::array_t<float,py::array::c_style|py::array::forcecast> values) {
        if(probs.ndim()!=2 || probs.shape(0)!=(int)ids.size() || probs.shape(1)!=A ||
           values.ndim()!=1 || values.shape(0)!=(int)ids.size() || ids.size()!=pending.size())
            throw std::runtime_error("Prediction shape mismatch");
        auto p=probs.unchecked<2>(); auto v=values.unchecked<1>();
        for(int i=0;i<(int)ids.size();i++) {
            if(!std::isfinite(v(i)) || std::abs(v(i))>1.0001) throw std::runtime_error("Invalid value");
            auto found=pending.find(ids[i]); if(found==pending.end()) throw std::runtime_error("Unknown request");
            auto& job=found->second; Node* leaf=job.path.back();
            double sum=0; int legal=0;
            for(int a=0;a<A;a++) if(!leaf->pos.board[a]) {
                if(!std::isfinite(p(i,a)) || p(i,a)<0) throw std::runtime_error("Invalid policy");
                sum+=p(i,a); legal++;
            }
            for(int a=0;a<A;a++) leaf->base_prior[a]=leaf->pos.board[a] ? 0 : (sum>0?p(i,a)/sum:1./legal);
            leaf->prior=leaf->base_prior;
            leaf->network_value=v(i);
            if(leaf==roots[job.tree].get()) noise(*leaf,job.tree);
            leaf->expanded=true; leaf->pending=false;
            backup(job.path,v(i),true);
            completed[job.tree]++; actual_evaluations++; simulations++;
            pending.erase(found);
        }
    }
    bool done() const {
        if(!pending.empty()) return false;
        for(int i=0;i<(int)roots.size();i++) if(completed[i]<targets[i]) return false;
        return true;
    }
    py::array_t<float> policy(double temperature,bool prune_forced=false) const {
        if(!done() || !std::isfinite(temperature) || temperature<0) throw std::runtime_error("Search incomplete or invalid temperature");
        py::array_t<float> result({(int)roots.size(),A}); auto p=result.mutable_unchecked<2>();
        for(int i=0;i<(int)roots.size();i++) {
            const Node& root=*roots[i]; double maxlog=-1e300;
            auto counts=visit_counts(i,prune_forced);
            for(int a=0;a<A;a++) {p(i,a)=0; if(counts[a]>0)
                maxlog=std::max(maxlog,std::log(counts[a]));}
            if(root.pos.terminal) continue;
            if(temperature==0) {
                // Return equal mass over maximum-count ties; caller uses seeded RNG.
                double maxn=*std::max_element(counts.begin(),counts.end());
                double sum=0;
                for(int a=0;a<A;a++) if(!root.pos.board[a] && maxn>0 && counts[a]==maxn) {p(i,a)=1;sum++;}
                if(sum==0) { for(int a=0;a<A;a++) {p(i,a)=root.prior[a];sum+=p(i,a);} }
                for(int a=0;a<A;a++) if(sum>0) p(i,a)/=sum;
            } else {
                double sum=0;
                for(int a=0;a<A;a++) if(counts[a]>0) {p(i,a)=std::exp((std::log(counts[a])-maxlog)/temperature);sum+=p(i,a);}
                if(sum==0) for(int a=0;a<A;a++) {p(i,a)=root.prior[a];sum+=p(i,a);}
                for(int a=0;a<A;a++) if(sum>0) p(i,a)/=sum;
            }
        }
        return result;
    }
    py::list root_stats() const {
        if(!done()) throw std::runtime_error("Search incomplete");
        py::list result;
        for(int i=0;i<(int)roots.size();i++) {
            const Node& root=*roots[i];auto raw=visit_counts(i,false), target=visit_counts(i,true);
            int visited=0,ties=0;double maxn=0,total=0,target_total=0;
            for(int a=0;a<A;a++) {
                visited+=raw[a]>0;maxn=std::max(maxn,raw[a]);total+=raw[a];target_total+=target[a];
            }
            double entropy=0,target_entropy=0;
            for(int a=0;a<A;a++) {
                ties+=maxn>0 && raw[a]==maxn;
                if(raw[a]>0) {double p=raw[a]/total;entropy-=p*std::log(p);}
                if(target[a]>0) {double p=target[a]/target_total;target_entropy-=p*std::log(p);}
                if(total==0 && !root.pos.terminal && root.prior[a]>0)
                    entropy-=root.prior[a]*std::log(root.prior[a]);
                if(target_total==0 && !root.pos.terminal && root.prior[a]>0)
                    target_entropy-=root.prior[a]*std::log(root.prior[a]);
            }
            py::dict row;
            row["visits"]=root.visits;row["visited_children"]=visited;row["max_visit_ties"]=ties;
            row["root_value"]=root.visits>0 ? root.total/root.visits : root.network_value;
            row["network_value"]=root.network_value;row["policy_entropy"]=entropy;row["target_entropy"]=target_entropy;
            row["max_visit_share"]=total>0 ? maxn/total : 0.;row["forced_playouts"]=root.forced_playouts;
            row["new_simulations"]=completed[i];row["max_depth"]=max_depth[i];
            row["mean_depth"]=completed[i]>0 ? (double)total_depth[i]/completed[i] : 0.;
            result.append(row);
        }
        return result;
    }
    py::tuple states() const {
        py::array_t<int8_t> boards({(int)roots.size(),A});auto b=boards.mutable_unchecked<2>();
        std::vector<int> colors,last,ply,winner;std::vector<bool> terminal;
        for(int i=0;i<(int)roots.size();i++) {
            auto& p=roots[i]->pos;
            for(int a=0;a<A;a++) b(i,a)=p.board[a];
            colors.push_back(p.color);last.push_back(p.last);ply.push_back(p.ply);winner.push_back(p.winner);terminal.push_back(p.terminal);
        }
        return py::make_tuple(boards,colors,last,ply,winner,terminal);
    }
    void advance(const std::vector<int>& actions) {
        if(!pending.empty() || actions.size()!=roots.size()) throw std::runtime_error("Invalid advance");
        for(int i=0;i<(int)roots.size();i++) {
            int a=actions[i];
            if(a<0) {if(!roots[i]->pos.terminal) throw std::runtime_error("Skip only terminal games");continue;}
            if(a>=A || roots[i]->pos.terminal || roots[i]->pos.board[a]) throw std::runtime_error("Illegal root advance");
            auto child=std::move(roots[i]->children[a]);
            if(!child) child=std::make_unique<Node>(roots[i]->pos.move(a));
            roots[i]=std::move(child);
        }
    }
    void set_history(int tree,const std::vector<int>& history) {
        if(!pending.empty() || tree<0 || tree>=(int)roots.size()) throw std::runtime_error("Invalid history reset");
        Position p; for(int a:history) p=p.move(a);
        roots[tree]=std::make_unique<Node>(p);
    }
    std::string rng_state() const {std::ostringstream out;out<<rng;return out.str();}
    void set_rng_state(const std::string& state) {std::istringstream in(state);in>>rng;if(!in)throw std::runtime_error("Invalid RNG state");}
    py::dict counters() const {py::dict d;d["network_evaluations"]=actual_evaluations;d["simulations"]=simulations;return d;}
    void clear_trees() {
        if(!pending.empty()) throw std::runtime_error("Outstanding requests");
        for(auto& root:roots) root=std::make_unique<Node>(root->pos);
    }
};

PYBIND11_MODULE(az_search,m) {
    py::class_<Forest>(m,"Forest")
        .def(py::init<int,unsigned long long>()).def("begin",&Forest::begin,
            py::arg("sims"),py::arg("cpuct"),py::arg("epsilon"),py::arg("alpha"),
            py::arg("fpu_reduction")=.2,py::arg("root_fpu_reduction")=.2,py::arg("forced_playouts")=0.,
            py::arg("budgets")=std::vector<int>{},py::arg("exploratory")=std::vector<bool>{})
        .def("request",&Forest::request).def("complete",&Forest::complete)
        .def("done",&Forest::done).def("policy",&Forest::policy,py::arg("temperature"),py::arg("prune_forced")=false)
        .def("root_stats",&Forest::root_stats)
        .def("states",&Forest::states).def("advance",&Forest::advance)
        .def("set_history",&Forest::set_history).def("rng_state",&Forest::rng_state)
        .def("set_rng_state",&Forest::set_rng_state).def("counters",&Forest::counters)
        .def("clear_trees",&Forest::clear_trees);
}
