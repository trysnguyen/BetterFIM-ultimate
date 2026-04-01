import random
import numpy as np
from utils.ic import run_IC
from utils.mf_dcv import calculate_MF_DCV
from utils.fitness import fitness_F
def local_search_hill_climbing(G, S, groups, ideal_influences, SN_scores,
                               p=0.01, mc=50, lamda=0.5,
                               max_iterations=20, max_no_improve=5):
    """
    Local search hill climbing với độ phức tạp thấp

    Args:
        G: đồ thị NetworkX
        S: seed set hiện tại (list)
        groups: dictionary {group_id: [nodes]}
        ideal_influences: dictionary {group_id: ideal_influence}
        SN_scores: PageRank scores của các nodes
        p: xác suất lan truyền IC
        mc: số lần Monte Carlo
        lamda: tham số lambda cho fitness
        max_iterations: số vòng lặp tối đa
        max_no_improve: số vòng lặp liên tiếp không cải thiện thì dừng

    Returns:
        best_S: seed set tốt nhất tìm được
        best_fit: fitness tốt nhất
        best_mf: MF tương ứng
        best_dcv: DCV tương ứng
    """
    # Đánh giá solution hiện tại
    current_S = list(S)
    current_mf, current_dcv = calculate_MF_DCV(G, current_S, groups, ideal_influences, p, mc)
    current_fit = fitness_F(current_mf, current_dcv, lamda)

    best_S = current_S.copy()
    best_fit = current_fit
    best_mf = current_mf
    best_dcv = current_dcv

    no_improve_count = 0
    all_nodes = list(G.nodes())

    for iteration in range(max_iterations):
        improved = False

        # Tạo danh sách candidates cho swap (nodes không trong S, có SN score cao)
        candidates = [n for n in all_nodes if n not in current_S]

        # Ưu tiên các nodes có SN score cao
        if len(candidates) > 100:  # Giới hạn số candidates để tăng tốc
            weights = np.array([SN_scores.get(n, 0) + 1e-8 for n in candidates])
            probs = weights / weights.sum()
            num_sample = min(100, len(candidates))
            candidates = list(np.random.choice(candidates, size=num_sample, replace=False, p=probs))

        # Thử swap từng node trong current_S
        # Chỉ thử một số nodes ngẫu nhiên thay vì toàn bộ để giảm complexity
        num_tries = min(5, len(current_S))  # Chỉ thử swap 5 nodes
        nodes_to_try = random.sample(current_S, num_tries)

        for node_out in nodes_to_try:
            # Thử swap với một số candidates ngẫu nhiên
            num_candidates_try = min(5, len(candidates))  # Chỉ thử 5 candidates
            candidates_to_try = random.sample(candidates, num_candidates_try)

            for node_in in candidates_to_try:
                # Tạo neighbor solution
                neighbor_S = current_S.copy()
                idx = neighbor_S.index(node_out)
                neighbor_S[idx] = node_in

                # Đánh giá neighbor
                neighbor_mf, neighbor_dcv = calculate_MF_DCV(G, neighbor_S, groups, ideal_influences, p, mc)
                neighbor_fit = fitness_F(neighbor_mf, neighbor_dcv, lamda)

                # Nếu tốt hơn, chấp nhận
                if neighbor_fit > current_fit:
                    current_S = neighbor_S
                    current_fit = neighbor_fit
                    current_mf = neighbor_mf
                    current_dcv = neighbor_dcv
                    improved = True

                    # Cập nhật best nếu cần
                    if current_fit > best_fit:
                        best_S = current_S.copy()
                        best_fit = current_fit
                        best_mf = current_mf
                        best_dcv = current_dcv

                    break  # Chuyển sang node tiếp theo

            if improved:
                break  # Chuyển sang iteration tiếp theo

        # Kiểm tra điều kiện dừng
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= max_no_improve:
            break

    return best_S, best_fit, (best_mf, best_dcv)