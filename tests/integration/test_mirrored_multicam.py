from eks.multicam_smoother import fit_eks_mirrored_multicam


def test_mirrored_multicam_defaults(compare_to_golden, tmp_path, pytestconfig, request):
    input_dir = str(pytestconfig.rootpath / 'data' / 'mirror-mouse')
    fit_eks_mirrored_multicam(
        input_source=input_dir,
        save_file=str(tmp_path / 'eks_mirrored_multicam.csv'),
        bodypart_list=['paw1LH', 'paw2LF'],
        camera_names=['top', 'bot'],
        quantile_keep_pca=95,
        inflate_vars=True,
    )
    compare_to_golden(request.node.name, tmp_path)


def test_mirrored_multicam_fixed_smooth_param(
    compare_to_golden, tmp_path, pytestconfig, request,
):
    input_dir = str(pytestconfig.rootpath / 'data' / 'mirror-mouse')
    fit_eks_mirrored_multicam(
        input_source=input_dir,
        save_file=str(tmp_path / 'eks_mirrored_multicam.csv'),
        bodypart_list=['paw1LH', 'paw2LF'],
        camera_names=['top', 'bot'],
        smooth_param=[10.0],
        quantile_keep_pca=95,
        inflate_vars=True,
    )
    compare_to_golden(request.node.name, tmp_path)
