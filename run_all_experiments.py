# run_all_experiments.py
# Author: Hyeong-Taek Kwon (Chosun Univ.)
# Part of the study: "A Comprehensive Survey on Preprocessing Strategies for RGB-Based 3D Gaussian Splatting"
import torch
from pathlib import Path
from hloc import (extract_features, match_features, 
                  reconstruction, pairs_from_exhaustive)

# 1. 경로 설정
ROOT_DIR = Path(__file__).parent
images_dir = ROOT_DIR / 'survey_data' / 'images'
base_output = ROOT_DIR / 'outputs' / 'survey_benchmark'

# 2. 서베이 논문 표(Table) 작성을 위한 12가지 교차 검증 매트릭스
experiments = [
    # --- Group A: SIFT Baseline ---
    ('01_SIFT_NN_Mutual', 'sift', 'NN-mutual'),
    ('02_SIFT_NN_Ratio', 'sift', 'NN-ratio'),
    ('03_SIFT_AdaLAM', 'sift', 'adalam'), # SIFT의 Scale 정보를 활용한 AdaLAM 필터링
    
    # --- Group B: Early Deep Features ---
    ('04_SOSNet_NN', 'sosnet', 'NN-mutual'),
    ('05_R2D2_NN', 'r2d2', 'NN-mutual'),
    ('06_D2Net_NN', 'd2net-ss', 'NN-mutual'), # GPU 에러 방지를 위해 CPU 우회 예정
    
    # --- Group C: SuperPoint Ecosystem (Ablation) ---
    ('07_SuperPoint_NN', 'superpoint_aachen', 'NN-superpoint'), # 딥러닝 특징점 + 전통 매칭
    ('08_SuperPoint_SuperGlue', 'superpoint_aachen', 'superglue'), # 딥러닝 + 딥러닝(GNN)
    ('09_SuperPoint_LightGlue', 'superpoint_aachen', 'superpoint+lightglue'), # 딥러닝 + 최신 Attention
    
    # --- Group D: Latest Features (DISK & ALIKED) ---
    ('10_DISK_NN', 'disk', 'NN-mutual'), # 최신 특징점 + 전통 매칭
    ('11_DISK_LightGlue', 'disk', 'disk+lightglue'), # 최신 + 최신 (SOTA 1)
    ('12_ALIKED_LightGlue', 'aliked-n16', 'aliked+lightglue'), # 최신 + 최신 (SOTA 2)
]

def run_benchmark():
    print(f"🚀 서베이 논문용 {len(experiments)}개 파이프라인 매트릭스 가동을 시작합니다...\n")
    
    for folder_name, f_key, m_key in experiments:
        print(f"{'='*55}")
        print(f"▶️ 진행 중: {folder_name} ({f_key} + {m_key})")
        print(f"{'='*55}")
        
        out_dir = base_output / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        sfm_dir = out_dir / 'sparse'
        pairs_path = out_dir / 'pairs.txt'
        
        f_conf = extract_features.confs.get(f_key)
        m_conf = match_features.confs.get(m_key)
        
        if not f_conf or not m_conf:
            print(f"⚠️ {f_key} 또는 {m_key} 설정이 없어 스킵합니다.")
            continue
            
        try:
            # 1. 특징점 추출 (D2Net 예외 처리 포함)
            if f_key == 'd2net-ss':
                print("   [안내] D2Net은 GPU 텐서 에러 방지를 위해 일시적으로 CPU 모드로 추출합니다.")
                orig_cuda_check = torch.cuda.is_available
                torch.cuda.is_available = lambda: False
                try:
                    feat_path = extract_features.main(f_conf, images_dir, out_dir)
                finally:
                    torch.cuda.is_available = orig_cuda_check # 원상복구
            else:
                feat_path = extract_features.main(f_conf, images_dir, out_dir)
            
            # 2. 매칭 쌍 생성
            pairs_from_exhaustive.main(pairs_path, features=feat_path)
            
            # 3. 매칭 수행
            match_path = match_features.main(m_conf, pairs_path, feat_path.stem, out_dir)
            
            # 4. SfM (초기 포인트 클라우드 생성)
            reconstruction.main(sfm_dir, images_dir, pairs_path, feat_path, match_path)
            
            print(f"✅ 성공: {folder_name} 완료!\n")
            
        except Exception as e:
            print(f"❌ 실패 ({folder_name}): {e}\n")
            continue

if __name__ == "__main__":
    run_benchmark()
