fit_memory.py -t G1 -o G1_tik_fit --log --norm --min --log_config -c tik.toml 
fit_memory.py -t G1 -o G1_adapt_fit --log --norm --min --log_config -c adapt.toml 
fit_memory.py -t G1 -o G1_scipy_fit --log --norm --min --log_config -c scipy.toml

fit_memory.py -t K1 -o K1_tik_fit --log --norm --min --log_config -c tik_k.toml
fit_memory.py -t K1 -o K1_adapt_fit --log --norm --min --log_config -c adapt_k.toml
fit_memory.py -t K1 -o K1_scipy_fit --log --norm --min --log_config -c scipy_k.toml

fit_memory.py -t G1 -o G1_tik_dosz_fit --log --norm --min --log_config -c tik_dosz.toml
fit_memory.py -t G1 -o G1_adapt_dosz_fit --log --norm --min --log_config -c adapt_dosz.toml
fit_memory.py -t G1 -o G1_scipy_dosz_fit --log --norm --min --log_config -c scipy_dosz.toml

fit_memory.py -t G1 -o G1_no_norm_fit --log  --min --log_config -c adapt.toml
fit_memory.py -t G1 -o G1_adapt_no_log_fit  --norm --min --log_config -c adapt.toml
fit_memory.py -t G1 -o G1_no_log_config_fit --log --norm --min  -c adapt.toml
fit_memory.py -t G1 -o G1_no_mim_fit --log --norm --log_cofig  -c adapt.toml
fit_memory.py -t G1 -o G1_minimal_output_fit -c adapt.toml




