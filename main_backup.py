import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from betterFIM_backup import betterFIM

if __name__ == "__main__":
    links_file = "dataset/synth3.links"
    attr_file = "dataset/synth3.attr"

    results = []
    mf_list = []
    dcv_list = []

    for i in range(20):
        print(datetime.now())
        print(f"Starting run {i + 1}/20...")
        result = betterFIM(links_file, attr_file)
        if result:
            fit, (mf, dcv), seed_set, IM = result
            results.append((fit, mf, dcv, seed_set))
            mf_list.append(mf)
            dcv_list.append(dcv)
            print(f"Run {i + 1} completed: F = {mf - dcv:.4f} (MF: {mf:.4f}, DCV: {dcv:.4f})")
            seed_set = [int(x) for x in seed_set]
            print(seed_set)
            print(IM)

    if results:
        # Tính F giống CEA-FIM: F = mean(MF) - mean(DCV)
        import numpy as np

        avg_mf = np.mean(mf_list)
        avg_dcv = np.mean(dcv_list)
        F_score = avg_mf - avg_dcv

        print("\n" + "=" * 60)
        print(f"Results over {len(results)} runs:")
        print(f"Average MF (Maximin Fairness): {avg_mf:.4f}")
        print(f"Average DCV (Diversity Constraints Violation): {avg_dcv:.4f}")
        print(f"F = mean(MF) - mean(DCV) = {F_score:.4f}")
        print("=" * 60)