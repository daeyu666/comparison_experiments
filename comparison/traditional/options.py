"""Torch-free root launcher arguments for registered-only classical fusion."""
DEFAULTS=dict(cnmf_iterations=200,cnmf_endmembers=0,hysure_iterations=200,hysure_rank=10,
              hysure_lambda_phi=5e-4,hysure_lambda_m=1.,hysure_cg_maxiter=80,
              hysure_cg_rtol=1e-6,classical_threads=1)


def add_classical_args(parser):
    for name,value in DEFAULTS.items():
        parser.add_argument("--"+name,type=type(value),default=value)
    parser.add_argument("--classical_save_predictions",action="store_true")


def benchmark_command(root,method,dataset,device,*,data_root,output_override="",
                      max_samples=0,eval_seed=1234,options=None):
    import sys
    if max_samples<0:
        raise ValueError("max_samples must be nonnegative")
    cmd=[sys.executable,str(root/"comparison"/method/"test.py"),
         "--dataset",dataset,"--mode","registered","--device",device,
         "--data_root",str(data_root),"--eval_seed",str(eval_seed),
         "--max_samples",str(max_samples)]
    if output_override:
        cmd+=["--output_dir",str(output_override)]
    options=options or {}
    for name,value in DEFAULTS.items():
        cmd+=["--"+name,str(options.get(name,value))]
    if options.get("classical_save_predictions",False):
        cmd.append("--classical_save_predictions")
    return cmd

