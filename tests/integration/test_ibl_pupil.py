from eks.ibl_pupil_smoother import fit_eks_pupil


def test_ibl_pupil_defaults(compare_to_golden, tmp_path, pytestconfig, request):
    input_dir = str(pytestconfig.rootpath / 'data' / 'ibl-pupil')
    fit_eks_pupil(
        input_source=input_dir,
        save_file=str(tmp_path / 'eks_ibl_pupil.csv'),
        smooth_params=[None, None],
    )
    compare_to_golden(request.node.name, tmp_path)


def test_ibl_pupil_fixed_smooth_param(compare_to_golden, tmp_path, pytestconfig, request):
    input_dir = str(pytestconfig.rootpath / 'data' / 'ibl-pupil')
    fit_eks_pupil(
        input_source=input_dir,
        save_file=str(tmp_path / 'eks_ibl_pupil.csv'),
        smooth_params=[0.99, 0.99],
    )
    compare_to_golden(request.node.name, tmp_path)
