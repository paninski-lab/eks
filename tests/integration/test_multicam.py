from eks.multicam_smoother import fit_eks_multicam


def test_multicam_defaults(compare_to_golden, tmp_path, pytestconfig, request):
    input_dir = str(pytestconfig.rootpath / 'data' / 'mirror-mouse-separate')
    fit_eks_multicam(
        input_source=input_dir,
        save_dir=str(tmp_path),
        bodypart_list=['paw1LH', 'paw2LF'],
        camera_names=['top', 'bot'],
        quantile_keep_pca=95,
        inflate_vars=True,
    )
    compare_to_golden(request.node.name, tmp_path)


def test_multicam_fixed_smooth_param(compare_to_golden, tmp_path, pytestconfig, request):
    input_dir = str(pytestconfig.rootpath / 'data' / 'mirror-mouse-separate')
    fit_eks_multicam(
        input_source=input_dir,
        save_dir=str(tmp_path),
        bodypart_list=['paw1LH', 'paw2LF'],
        camera_names=['top', 'bot'],
        smooth_param=[10.0],
        quantile_keep_pca=95,
        inflate_vars=True,
    )
    compare_to_golden(request.node.name, tmp_path)


def test_multicam_defaults_nonlinear(compare_to_golden, tmp_path, pytestconfig, request):
    input_dir = str(pytestconfig.rootpath / 'data' / 'fly')
    fit_eks_multicam(
        input_source=input_dir,
        save_dir=str(tmp_path),
        bodypart_list=['L1A', 'L1B'],
        calibration=str(pytestconfig.rootpath / 'data' / 'fly' / 'calibration.toml'),
        quantile_keep_pca=95,
        inflate_vars=True,
    )
    compare_to_golden(request.node.name, tmp_path)


def test_multicam_fixed_smooth_param_nonlinear(
    compare_to_golden, tmp_path, pytestconfig, request,
):
    input_dir = str(pytestconfig.rootpath / 'data' / 'fly')
    fit_eks_multicam(
        input_source=input_dir,
        save_dir=str(tmp_path),
        bodypart_list=['L1A', 'L1B'],
        calibration=str(pytestconfig.rootpath / 'data' / 'fly' / 'calibration.toml'),
        smooth_param=[10.0],
        quantile_keep_pca=95,
        inflate_vars=True,
    )
    compare_to_golden(request.node.name, tmp_path)
