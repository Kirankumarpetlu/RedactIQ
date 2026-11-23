from scripts.run_first_500_benchmark import default_config_for_device


def test_first500_cuda_uses_measured_latency_profile():
    assert default_config_for_device("cuda:0").name == "pipeline_config.balanced_latency.yaml"
    assert default_config_for_device("cuda").name == "pipeline_config.balanced_latency.yaml"


def test_first500_cpu_retains_portable_profile():
    assert default_config_for_device("cpu").name == "pipeline_config.yaml"
    assert default_config_for_device("auto").name == "pipeline_config.yaml"
