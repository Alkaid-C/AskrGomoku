#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// ============================================================================
// MCTSNode
// ============================================================================
//
// Child statistics are stored as contiguous std::vector<float/int> so the
// PUCT inner loop in select_child runs with native CPU instructions and
// cache-friendly memory access — no Python interpreter overhead per element.
//
// Ownership: each non-root node is owned (via shared_ptr) by its parent's
// child_node vector. The parent pointer is a raw non-owning pointer; it is
// always valid during backup because backup is called inside the simulation
// loop while the root (and therefore the entire tree) is still live.

class MCTSNode : public std::enable_shared_from_this<MCTSNode> {
public:
    // Tree linkage (raw parent pointer: non-owning, always valid during sim)
    MCTSNode* parent_raw;
    int parent_k;
    int action;
    float prior;

    // Child statistics — populated by expand()
    std::vector<int>   child_actions;
    std::vector<float> child_priors;
    std::vector<float> child_q;
    std::vector<int>   child_n;
    std::vector<float> child_total;
    std::vector<std::shared_ptr<MCTSNode>> child_node;  // null = not yet materialized

    // Node state
    bool  is_expanded;
    bool  is_terminal;
    float terminal_value;
    int   visit_count;

    // Root constructor (called from Python)
    MCTSNode()
        : parent_raw(nullptr), parent_k(-1), action(-1), prior(0.0f),
          is_expanded(false), is_terminal(false), terminal_value(0.0f),
          visit_count(0)
    {}

    // Child constructor (called from select_child only)
    MCTSNode(MCTSNode* parent, int pk, int act, float pr)
        : parent_raw(parent), parent_k(pk), action(act), prior(pr),
          is_expanded(false), is_terminal(false), terminal_value(0.0f),
          visit_count(0)
    {}

    // Populate child arrays and mark node as expanded.
    void expand(const std::vector<int>& actions, const std::vector<float>& priors) {
        int n = static_cast<int>(actions.size());
        child_actions = actions;
        child_priors  = priors;
        child_q.assign(n, 0.0f);
        child_n.assign(n, 0);
        child_total.assign(n, 0.0f);
        child_node.assign(n, nullptr);
        is_expanded = true;
    }

    // PUCT child selection. Materializes the chosen child node on demand.
    std::shared_ptr<MCTSNode> select_child(float c_puct) {
        float c_sqrt = c_puct * std::sqrt(static_cast<float>(visit_count));
        int K = static_cast<int>(child_actions.size());
        int best_k = -1;
        float best_score = -std::numeric_limits<float>::infinity();

        for (int k = 0; k < K; ++k) {
            float score = child_q[k]
                        + c_sqrt * child_priors[k] / (1.0f + static_cast<float>(child_n[k]));
            if (score > best_score) {
                best_score = score;
                best_k = k;
            }
        }

        if (best_k < 0)
            throw std::runtime_error("select_child: node has no children");

        if (!child_node[best_k]) {
            child_node[best_k] = std::make_shared<MCTSNode>(
                this, best_k, child_actions[best_k], child_priors[best_k]
            );
        }
        return child_node[best_k];
    }

    // Walk from this node to root, flipping perspective and discounting each step.
    void backup(float value, float gamma) {
        float v = value;
        MCTSNode* node = this;
        while (node != nullptr) {
            v = -v * gamma;
            node->visit_count += 1;
            if (node->parent_raw != nullptr) {
                int k = node->parent_k;
                node->parent_raw->child_total[k] += v;
                node->parent_raw->child_n[k]     += 1;
                node->parent_raw->child_q[k]      =
                    node->parent_raw->child_total[k]
                    / static_cast<float>(node->parent_raw->child_n[k]);
            }
            node = node->parent_raw;
        }
    }
};

// ============================================================================
// pybind11 bindings
// ============================================================================

PYBIND11_MODULE(mcts_ext, m) {
    m.doc() = "C++ MCTS node with select_child and backup";

    py::class_<MCTSNode, std::shared_ptr<MCTSNode>>(m, "MCTSNode")
        .def(py::init<>())
        .def("expand",       &MCTSNode::expand,       py::arg("actions"), py::arg("priors"))
        .def("select_child", &MCTSNode::select_child, py::arg("c_puct"))
        .def("backup",       &MCTSNode::backup,       py::arg("value"), py::arg("gamma"))
        // Attributes read/written by the Python simulation loop
        .def_readwrite("action",         &MCTSNode::action)
        .def_readwrite("visit_count",    &MCTSNode::visit_count)
        .def_readwrite("is_expanded",    &MCTSNode::is_expanded)
        .def_readwrite("is_terminal",    &MCTSNode::is_terminal)
        .def_readwrite("terminal_value", &MCTSNode::terminal_value)
        // Read at the end of search to extract visit distributions and root Q
        .def_readwrite("child_actions",  &MCTSNode::child_actions)
        .def_readwrite("child_n",        &MCTSNode::child_n)
        .def_readwrite("child_q",        &MCTSNode::child_q)
    ;
}
