"""Read-only checkpoint audit; write findings to a new external directory."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'experiments/src'))
import torch
from lift_ce.models.vision import SmallCNN, StrongConvAutoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path = Path('/mnt/big_disk/xu.zhao/robust_ce_results/mnist_roar_latent_calibrated/payload.pt')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    model, ae = SmallCNN().eval(), StrongConvAutoencoder(64).eval()
    model.load_state_dict(payload['model'], strict=True)
    ae.load_state_dict(payload['ae'], strict=True)
    x = payload['sources']
    with torch.inference_mode():
        reconstructed = ae(x)
        metrics = {'source_count': len(x), 'reconstruction_mse': ((x-reconstructed)**2).mean().item(),
                   'classifier_decision_agreement': ((model(x)>0)==(model(reconstructed)>0)).float().mean().item()}
    def command(argv):
        p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {'argv': argv, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    hardware = {'hostname': socket.gethostname(), 'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'GENV_GPUS': os.environ.get('GENV_GPUS'), 'requested_gpu_count': 7,
                'selected_gpu_ids': None,
                'physical_gpus': command(['/usr/bin/nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu','--format=csv']),
                'running_processes': command(['/usr/bin/nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'])}
    role = {'status': 'awaiting_task_identity', 'observed_task': {'source_digit':4,'target_digit':9},
            'evidence': '/home/xu.zhao/robust_ce/configs/mnist_strong_generator_visual_4_to_9.yaml',
            'classifier_4_through_9_replicates': 'not found; 4_to_9 is a digit pair in discovered source',
            'candidate_checkpoint': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'classifier_key': 'model', 'classifier_architecture': 'SmallCNN(embedding_dim=128), binary logit',
            'autoencoder_key': 'ae', 'autoencoder_architecture': 'StrongConvAutoencoder(latent_dim=64)',
            'selected_f0': None, 'seed': 'source CLI default 0; original command not recovered'}
    for name, obj in [('hardware_manifest.json',hardware),('classifier_role_manifest.json',role),('checkpoint_sanity.json',metrics),
                      ('resolved_config.json',{'status':'discovery_only','beta_geo':[1.0,1.05,1.1,1.2],
                       'current_validity_repair':False,'risk_aggregation':'max_model mean_CF hinge',
                       'task_identity_pending':True,'gpu_allocation_pending':True,'experiment_launched':False})]:
        (args.output/name).write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'sanity':metrics},indent=2))

if __name__ == '__main__':
    main()
