import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from collections import defaultdict
from prompt import PROMPTS
from utils import generate, match_answer, extract_between_braces
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

random.seed(42)


@dataclass
class ReasoningStep:
    """Represents a single step in student's reasoning process"""
    step_number: int
    description: str
    result: Any
    desc_embedding: Optional[List[float]] = None
    is_error: bool = False
    error_type: Optional[str] = None
    is_final_answer: bool = False


@dataclass
class StudentRecord:
    """Represents a training record with problem and student's answer"""
    problem_stem: str
    student_answer: Any
    correct_answer: Optional[Any] = None


class LLMInterface(ABC):
    """Abstract interface for LLM interactions"""
    def __init__(self, file_path: str = None, model: str = "gpt-4o"):
        self.file_path = file_path
        self.model = model
    
    @abstractmethod
    def generate_next_reasoning_step(self, problem: str, previous_steps: List[ReasoningStep], 
                                   num_candidates: int = 3, include_errors: bool = True) -> List[ReasoningStep]:
        """Generate possible next reasoning steps (both correct and incorrect)"""
        pass
    
    @abstractmethod
    def simulate_to_completion(self, problem: str, current_steps: List[ReasoningStep]) -> Any:
        """Simulate reasoning to completion assuming no further errors"""
        pass

class MCTSNode:
    """Node in the MCTS tree representing a partial reasoning path"""
    
    def __init__(self, id: str = None, is_terminal: bool = False, is_final_answer: bool = False,   reasoning_steps: List[ReasoningStep] = None, parent: Optional['MCTSNode'] = None, answer: Any = None):
        self.id = id
        self.reasoning_steps = reasoning_steps
        self.parent = parent

        if is_terminal:
            self.children: List['MCTSNode'] = []
        else:
            self.children: List['MCTSNode'] = [MCTSNode(id=f"{self.id}-0", is_terminal=True, is_final_answer=True, reasoning_steps=reasoning_steps, parent=self, answer=answer)]
        
        self.visits = 0
        self.total_reward = 0.0
        self.is_terminal = is_terminal
        self.is_final_answer = is_final_answer
        self.answer = answer
        self.is_expanded = False
        
    def uct_value(self, c=1.414) -> float:
        """Calculate UCT (Upper Confidence Bound for Trees) value"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, c=1.414) -> 'MCTSNode':
        """Select best child based on UCT value"""
        return max(self.children[1:], key=lambda child: child.uct_value(c))
    
    def add_child(self, idx: int, reasoning_step: ReasoningStep) -> 'MCTSNode':
        """Add a new child node with an additional reasoning step"""
        new_steps = self.reasoning_steps + [reasoning_step]
        answer = reasoning_step.result
        if type(answer) == dict or type(answer) == list:
            answer = json.dumps(answer)
        elif type(answer) == int or type(answer) == float or type(answer) == bool:
            answer = str(answer)
        elif answer is None:
            answer = "None"
        if type(answer) != str:
            print(answer)
            print(type(answer))
        assert type(answer) == str
        child = MCTSNode(id=f"{self.id}-{idx}", reasoning_steps=new_steps, parent=self, answer=answer)
        if reasoning_step.is_final_answer:
            child.is_final_answer = True
        self.children.append(child)
        return child
    
    def update(self, reward: float, update_visits: bool = True):
        """Update node statistics with backpropagation"""
        if update_visits:
            self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.update(reward, update_visits)

def visualize_mcts_tree(root: MCTSNode, record: StudentRecord, 
                       filename: Optional[str] = None, 
                       show_details: bool = True,
                       max_depth: Optional[int] = None) -> None:
    """
    Visualize the MCTS tree showing reasoning paths
    
    Args:
        root: Root node of the MCTS tree
        record: Student record with problem and answer
        filename: If provided, save the visualization to file
        show_details: Whether to show detailed information in nodes
        max_depth: Maximum depth to visualize (None for full tree)
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Track node positions and information
    node_info = {}
    node_positions = {}
    
    # BFS to build the graph
    queue = [(root, 0, 0, "root")]
    node_id = 0
    level_width = defaultdict(int)
    
    while queue:
        node, depth, sibling_index, parent_id = queue.pop(0)
        
        if max_depth and depth > max_depth:
            continue
            
        current_id = f"node_{node_id}"
        node_id += 1
        
        # Add node to graph
        G.add_node(current_id)
        if parent_id != "root":
            G.add_edge(parent_id, current_id)
        
        # Calculate node information
        avg_reward = node.total_reward / node.visits if node.visits > 0 else 0
        
        # Get step description
        if node.reasoning_steps:
            last_step = node.reasoning_steps[-1]
            step_desc = last_step.description[:30] + "..." if len(last_step.description) > 30 else last_step.description
            is_error = last_step.is_error
            error_type = last_step.error_type
        else:
            step_desc = "START"
            is_error = False
            error_type = None
        
        node_info[current_id] = {
            'id': node.id,
            'visits': node.visits,
            'avg_reward': avg_reward,
            'total_reward': node.total_reward,
            'step_desc': step_desc,
            'is_error': is_error,
            'error_type': error_type,
            'is_terminal': node.is_terminal,
            'is_final_answer': node.is_final_answer,
            'answer': node.answer,
            'depth': depth
        }
        
        # Add children to queue
        for i, child in enumerate(node.children):
            queue.append((child, depth + 1, i, current_id))
            level_width[depth + 1] += 1
    
    # Create layout using hierarchical positioning
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot') if nx.nx_agraph else nx.spring_layout(G)
    
    # Create figure
    plt.figure(figsize=(20, 12))
    ax = plt.gca()
    
    # Draw edges with varying thickness based on visits
    edge_visits = []
    for edge in G.edges():
        parent_visits = node_info[edge[0]]['visits']
        edge_visits.append(parent_visits)
    
    max_visits = max(edge_visits) if edge_visits else 1
    edge_widths = [3 * (v / max_visits) + 0.5 for v in edge_visits]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, 
                          edge_color='gray', alpha=0.6, 
                          arrows=True, arrowsize=10)
    
    # Draw nodes with colors based on average reward and error status
    for node_id, info in node_info.items():
        x, y = pos[node_id]
        
        # Determine node color
        if info['is_error']:
            color = '#ffcccc'  # Light red for errors
            edge_color = 'red'
        elif info['is_terminal']:
            if info['avg_reward'] > 0:
                color = '#ccffcc'  # Light green for successful terminal
                edge_color = 'green'
            else:
                color = '#ffcccc'  # Light red for failed terminal
                edge_color = 'red'
        else:
            # Color based on average reward
            if info['avg_reward'] > 0:
                color = '#e6f3ff'  # Light blue for positive
                edge_color = 'blue'
            else:
                color = '#fff0e6'  # Light orange for negative
                edge_color = 'orange'
        
        # Node size based on visits
        node_size = max(300, min(2000, 300 + info['visits'] * 20))
        
        # Draw node
        circle = plt.Circle((x, y), node_size/8000, color=color, 
                           ec=edge_color, linewidth=2, zorder=2)
        ax.add_patch(circle)
        
        # Add text
        if show_details:
            text_lines = [info['step_desc']]
            text_lines.append(f"V:{info['visits']} R:{info['avg_reward']:.2f}")
            if info['is_error']:
                text_lines.append(f"ERR: {info['error_type']}")
            if info['is_terminal']:
                text_lines.append(f"ANS: {info['answer']}")
            
            text = '\n'.join(text_lines)
            fontsize = max(6, min(10, 8 - info['depth']))
        else:
            text = f"{info['visits']}\n{info['avg_reward']:.2f}"
            fontsize = 8
        
        plt.text(x, y, text, ha='center', va='center', 
                fontsize=fontsize, weight='bold' if info['visits'] > 10 else 'normal')
    
    # Add title and problem information
    plt.title(f"MCTS Reasoning Tree\nProblem: {record.problem_stem}\nStudent Answer: {record.student_answer}", 
             fontsize=14, pad=20)
    
    # Add legend
    legend_elements = [
        patches.Patch(color='#ccffcc', label='Successful Path'),
        patches.Patch(color='#ffcccc', label='Error/Failed Path'),
        patches.Patch(color='#e6f3ff', label='Positive Reward'),
        patches.Patch(color='#fff0e6', label='Negative Reward'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add annotation for best path
    best_path_nodes = _get_best_path_nodes(root)
    best_path_ids = []
    
    # Map nodes to IDs
    queue = [(root, "node_0")]
    node_counter = 1
    while queue and len(best_path_ids) < len(best_path_nodes):
        node, node_id = queue.pop(0)
        if node in best_path_nodes:
            best_path_ids.append(node_id)
        for child in node.children:
            queue.append((child, f"node_{node_counter}"))
            node_counter += 1
    
    # Highlight best path
    if len(best_path_ids) > 1:
        best_path_edges = [(best_path_ids[i], best_path_ids[i+1]) 
                          for i in range(len(best_path_ids)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=best_path_edges,
                             width=4, edge_color='darkgreen', alpha=0.8)
    
    plt.axis('off')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def _get_best_path_nodes(root: MCTSNode) -> List[MCTSNode]:
    """Get nodes along the best path"""
    path = [root]
    current = root
    
    while current.children:
        best_child = max(current.children, 
                        key=lambda c: c.total_reward / c.visits if c.visits > 0 else 0)
        path.append(best_child)
        current = best_child
    
    return path


def visualize_reasoning_path(reasoning_path: List[ReasoningStep], 
                           record: StudentRecord,
                           filename: Optional[str] = None) -> None:
    """
    Visualize a specific reasoning path as a flowchart
    
    Args:
        reasoning_path: List of reasoning steps
        record: Student record with problem and answer  
        filename: If provided, save the visualization to file
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 2 + len(reasoning_path) * 1.5))
    
    # Starting y position
    y_start = len(reasoning_path) + 1
    x_center = 0.5
    
    # Draw problem at top
    problem_box = FancyBboxPatch((0.1, y_start), 0.8, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='lightblue',
                                edgecolor='darkblue',
                                linewidth=2)
    ax.add_patch(problem_box)
    ax.text(x_center, y_start + 0.4, f"Problem: {record.problem_stem}",
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw each reasoning step
    y_current = y_start - 1.5
    for i, step in enumerate(reasoning_path):
        # Determine box color based on error
        if step.is_error:
            facecolor = '#ffcccc'
            edgecolor = 'red'
        else:
            facecolor = '#ccffcc'
            edgecolor = 'green'
        
        # Draw box
        box = FancyBboxPatch((0.1, y_current - 0.4), 0.8, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=facecolor,
                            edgecolor=edgecolor,
                            linewidth=2)
        ax.add_patch(box)
        
        # Add text
        text = f"Step {step.step_number}: {step.description}"
        if step.result:
            text += f"\nResult: {step.result}"
        if step.is_error:
            text += f"\n(Error: {step.error_type})"
        
        ax.text(x_center, y_current, text,
                ha='center', va='center', fontsize=9)
        
        # Draw arrow
        if i < len(reasoning_path) - 1:
            ax.arrow(x_center, y_current - 0.5, 0, -0.5,
                    head_width=0.05, head_length=0.1, fc='black', ec='black')
        
        y_current -= 1.5
    
    # Draw final answer box
    y_current += 0.5
    answer_box = FancyBboxPatch((0.1, y_current - 0.4), 0.8, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor='lightyellow',
                               edgecolor='orange',
                               linewidth=2)
    ax.add_patch(answer_box)
    ax.text(x_center, y_current, f"Student Answer: {record.student_answer}",
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Set axis properties
    ax.set_xlim(0, 1)
    ax.set_ylim(y_current - 1, y_start + 1.5)
    ax.axis('off')
    
    plt.title("Student Reasoning Path", fontsize=14, pad=20)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()



def search_for_each_distractor(model, problem, distractor_index, file_dir, max_iterations=10):
    if os.path.isfile(f"{file_dir}/reasoning_paths_{max_iterations}.json"):
        with open(f"{file_dir}/reasoning_paths_{max_iterations}.json", "r") as f:
            for line in f.readlines():
                line = json.loads(line)
                if line["id"] == f"{problem['problem_id']}-{distractor_index}":
                    return line
    record = StudentRecord(
        problem_stem=problem["stem"],
        student_answer=problem["choices"][distractor_index],
        correct_answer=problem["choices"][problem["answer"]]
    )
    llm = MockLLM(file_path=f"{file_dir}/{problem['problem_id']}_{distractor_index}.jsonl", model=model)
    mcts = StudentReasoningMCTS(llm, max_iterations=max_iterations)
    reasoning_path, answer, root = mcts.search(record)

    line = {
        "id": f"{problem['problem_id']}-{distractor_index}",
        "stem": problem["stem"],
        "distractor": problem["choices"][distractor_index],
        "reasoning_path": reasoning_path,
        "answer": answer
    }
    with open(f"{file_dir}/reasoning_paths_{max_iterations}.json", "a") as f:
        f.write(json.dumps(line, ensure_ascii=False)+"\n")
    return line