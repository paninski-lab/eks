from eks.ibl_paw_multicam_smoother import fit_eks_multicam_ibl_paw


def test_ibl_paw_defaults(compare_to_golden, tmp_path, pytestconfig, request):
    input_dir = str(pytestconfig.rootpath / 'data' / 'ibl-paw')
    fit_eks_multicam_ibl_paw(
        input_source=input_dir,
        save_dir=str(tmp_path),
        var_mode='var',
        quantile_keep_pca=95,
        inflate_vars=True,
    )
    compare_to_golden(request.node.name, tmp_path)


def test_ibl_paw_fixed_smooth_param(compare_to_golden, tmp_path, pytestconfig, request):
    input_dir = str(pytestconfig.rootpath / 'data' / 'ibl-paw')
    fit_eks_multicam_ibl_paw(
        input_source=input_dir,
        save_dir=str(tmp_path),
        smooth_param=[10.0],
        var_mode='var',
        quantile_keep_pca=95,
        inflate_vars=True,
    )
    compare_to_golden(request.node.name, tmp_path)
