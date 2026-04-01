import random
import numpy as np
import copy
from utils.ic import run_IC
from utils.mf_dcv import calculate_MF_DCV
from utils.fitness import fitness_F


def local_search_hill_climbing(G, S, groups, ideal_influences, SN_scores,
                               p=0.01, mc=200, lamda=0.5,
                               max_iterations=5, top_k=5):
    """
    Local search dựa trên local_search_in_loop - đơn giản và hiệu quả
    
    Args:
        G: đồ thị NetworkX
        S: seed set hiện tại (list)
        groups: dictionary {group_id: [nodes]}
        ideal_influences: dictionary {group_id: ideal_influence}
        SN_scores: PageRank scores của các nodes (tương đương prank)
        p: xác suất lan truyền IC
        mc: số lần Monte Carlo
        lamda: tham số lambda cho fitness
        max_iterations: số vòng lặp tối đa (mặc định 10)
        top_k: số lượng top candidates để thử (mặc định 5)
    
    Returns:
        best_S: seed set tốt nhất tìm được
        best_fit: fitness tốt nhất
        best_metrics: (best_mf, best_dcv)
    """
    # Hàm Eval để đánh giá fitness
    def Eval(solution):
        mf, dcv = calculate_MF_DCV(G, solution, groups, ideal_influences, p, mc)
        fit = fitness_F(mf, dcv, lamda)
        return fit, mf, dcv
    
    # Khởi tạo best solution
    best_solution = copy.deepcopy(S)
    best_fitness, best_mf, best_dcv = Eval(best_solution)
    
    # Danh sách tất cả các nodes (tương đương partition.keys())
    all_nodes = list(G.nodes())
    
    # Vòng lặp chính
    for iteration in range(max_iterations):
        improved = False
        
        # Lấy top_k candidates có SN_scores cao nhất (tương đương prank)
        candidates = [n for n in all_nodes if n not in best_solution]
        candidates.sort(key=lambda x: SN_scores.get(x, 0), reverse=True)
        top_candidates = candidates[:top_k]
        
        # Thử thay thế từng vị trí trong solution
        for i in range(len(best_solution)):
            for candidate in top_candidates:
                # Tạo test solution
                test_solution = copy.deepcopy(best_solution)
                test_solution[i] = candidate
                
                # Đánh giá fitness
                test_fitness, test_mf, test_dcv = Eval(test_solution)
                
                # Nếu tốt hơn, cập nhật
                if test_fitness > best_fitness:
                    best_fitness = test_fitness
                    best_mf = test_mf
                    best_dcv = test_dcv
                    best_solution = test_solution
                    improved = True
                    break  # Break khỏi vòng lặp candidates
            
            if improved:
                break  # Break khỏi vòng lặp positions
        
        # Nếu không cải thiện được, dừng sớm
        if not improved:
            break
    
    return best_solution, best_fitness, (best_mf, best_dcv)

def local_search_end(G, S, groups, ideal_influences, SN_scores,
                               p=0.01, mc=200, lamda=0.5,
                               max_iterations=40, top_k=10):
    """
    Local search dựa trên local_search_in_loop - đơn giản và hiệu quả
    
    Args:
        G: đồ thị NetworkX
        S: seed set hiện tại (list)
        groups: dictionary {group_id: [nodes]}
        ideal_influences: dictionary {group_id: ideal_influence}
        SN_scores: PageRank scores của các nodes (tương đương prank)
        p: xác suất lan truyền IC
        mc: số lần Monte Carlo
        lamda: tham số lambda cho fitness
        max_iterations: số vòng lặp tối đa (mặc định 10)
        top_k: số lượng top candidates để thử (mặc định 5)
    
    Returns:
        best_S: seed set tốt nhất tìm được
        best_fit: fitness tốt nhất
        best_metrics: (best_mf, best_dcv)
    """
    # Hàm Eval để đánh giá fitness
    def Eval(solution):
        mf, dcv = calculate_MF_DCV(G, solution, groups, ideal_influences, p, mc)
        fit = fitness_F(mf, dcv, lamda)
        return fit, mf, dcv
    
    # Khởi tạo best solution
    best_solution = copy.deepcopy(S)
    best_fitness, best_mf, best_dcv = Eval(best_solution)
    
    # Danh sách tất cả các nodes (tương đương partition.keys())
    all_nodes = list(G.nodes())
    
    # Vòng lặp chính
    for iteration in range(max_iterations):
        improved = False
        
        # Lấy top_k candidates có SN_scores cao nhất (tương đương prank)
        candidates = [n for n in all_nodes if n not in best_solution]
        candidates.sort(key=lambda x: SN_scores.get(x, 0), reverse=True)
        top_candidates = candidates[:top_k]
        
        # Thử thay thế từng vị trí trong solution
        for i in range(len(best_solution)):
            for candidate in top_candidates:
                # Tạo test solution
                test_solution = copy.deepcopy(best_solution)
                test_solution[i] = candidate
                
                # Đánh giá fitness
                test_fitness, test_mf, test_dcv = Eval(test_solution)
                
                # Nếu tốt hơn, cập nhật
                if test_fitness > best_fitness:
                    best_fitness = test_fitness
                    best_mf = test_mf
                    best_dcv = test_dcv
                    best_solution = test_solution
                    improved = True
                    break  # Break khỏi vòng lặp candidates
            
            if improved:
                break  # Break khỏi vòng lặp positions
        
        # Nếu không cải thiện được, dừng sớm
        if not improved:
            break
    
    return best_solution, best_fitness, (best_mf, best_dcv)