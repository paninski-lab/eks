from eks.singlecam_smoother import fit_eks_singlecam


def test_singlecam_defaults(compare_to_golden, tmp_path, pytestconfig, request):
    input_dir = str(pytestconfig.rootpath / 'data' / 'ibl-pupil')
    fit_eks_singlecam(
        input_source=input_dir,
        save_file=str(tmp_path / 'eks_singlecam.csv'),
    )
    compare_to_golden(request.node.name, tmp_path)


def test_singlecam_fixed_smooth_param(compare_to_golden, tmp_path, pytestconfig, request):
    input_dir = str(pytestconfig.rootpath / 'data' / 'ibl-pupil')
    fit_eks_singlecam(
        input_source=input_dir,
        save_file=str(tmp_path / 'eks_singlecam.csv'),
        smooth_param=[10.0],
    )
    compare_to_golden(request.node.name, tmp_path)
