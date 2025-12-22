from typing import Union, Optional                                                                                                                                                                                                                                              
                                                                                                   
import torch                                                                                       
from torch_geometric.data import Data                                                              
from torch_geometric.utils import to_networkx, scatter                                             
                                                                                                   
import networkx as nx                                                                              
                                                                                                   
def draw_graph(data: Data, with_labels: bool = True):                                              
    """Draw a graph using NetworkX heuristic-based algorithm.                                      
                                                                                                   
    Args:                                                                                          
        data: The graph to be drawn.                                                               
    """                                                                                            
    graph = to_networkx(data)                                                                      
    nx.draw(graph, with_labels=with_labels)                                                        
                                                                                                   
def laplacian(data: Data,                                                                          
              edge_weight: Optional[torch.Tensor] = None,                                          
              dtype: Optional[torch.dtype] = None,                                                 
              normalization: Optional[str] = None                                                  
     ):                                                                                            
    """Compute the graph Laplacian for simple graphs.                                              
                                                                                                   
    TODO: test for weighted graphs.                                                                
    TODO: implement normalization                                                                  
                                                                                                   
    """                                                                                            
                                                                                                   
    data.validate()                                                                                
                                                                                                   
    edge_index = data.edge_index                                                                   
                                                                                                   
    filtered_data = remove_self_loops(data)                                                        
    if edge_weight is None:                                                                        
        edge_weight = torch.ones(data.edge_index.size(dim=1), dtype=dtype,                         
                                 device=edge_index.device)                                         
                                                                                                   
    num_nodes, _ = data.x.size()                                                                   
    rows, cols = edge_index                                                                        
    degree = scatter(edge_weight, rows, dim=0, dim_size=num_nodes)                                 
                                                                                                   
    # L = D - A                                                                                    
    laplacian_data = add_self_loops(filtered_data, dtype=dtype)                                    
    laplacian_edge_weight = torch.cat([-edge_weight, degree], dim=0)                               
                                                                                                   
    return laplacian_data, laplacian_edge_weight   

def remove_self_loops(data: Data):                                                              
    """Remove self-edges from graph.                                                            
                                                                                                
    Args:                                                                                       
        data: The graph to be drawn.                                                            
                                                                                                
    Returns:                                                                                    
        filtered_data: A copy of the original graph                                             
                                                                                                
    """                                                                                         
    edge_index = data.edge_index                                                                
    try:                                                                                        
        edge_attr = data.edge_attr                                                              
    except AttributeError:                                                                      
        edge_attr = None                                                                        
                                                                                                
    mask = (edge_index[0] != edge_index[1])                                                     
                                                                                                
    filtered_data = data.clone()                                                                
    filtered_data.edge_index = edge_index[:, mask]                                              
    if edge_attr is not None:                                                                   
        filtered_data.edge_attr = edge_attr[:, mask]                                            
                                                                                                
    return filtered_data                                                                        
                                                                                                
def add_self_loops(data: Data, fill_value: Optional[Union[torch.Tensor, float]] = None,         
                  dtype: Optional[torch.dtype] = None):                                         
    """Add self-edges to graph. Assumes that no self-edges exist.                               
                                                                                                
    Args:                                                                                       
        data: The graph to be drawn.                                                            
                                                                                                
    Returns:                                                                                    
        filtered_data: A copy of the original graph                                             
                                                                                                
    """                                                                                         
    edge_index = data.edge_index                                                                
    try:                                                                                        
        edge_attr = data.edge_attr                                                              
    except AttributeError:                                                                      
        edge_attr = None                                                                        
                                                                                                
    num_nodes, _ = data.x.size()                                                                
    self_loops = torch.tile(torch.arange(num_nodes, dtype=dtype), (2, 1))                       
                                                                                                
    augmented_data = data.clone()                                                               
    augmented_data.edge_index = torch.cat([edge_index, self_loops], dim=1)                      
    if edge_attr is not None:                                                                   
        assert fill_value.dtype == dtype                                                        
        self_attr = torch.tile(fill_value, (1, num_nodes))                                      
        augmented_data.edge_attr = torch.cat([edge_attr, self_attr], dim=1)                     
                                                                                                
    return augmented_data                                                                                                                                                                                                                                                       
