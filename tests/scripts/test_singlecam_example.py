

def test_singlecam_example_defaults(
    run_script, compare_to_golden, tmpdir, pytestconfig, request,
):

    output_dir = run_script(
        script_file=str(pytestconfig.rootpath / 'scripts' / 'singlecam_example.py'),
        input_dir=str(pytestconfig.rootpath / 'data/ibl-pupil'),
        output_dir=tmpdir,
    )
    compare_to_golden(request.node.name, output_dir)


def test_singlecam_example_fixed_smooth_param(
    run_script, compare_to_golden, tmpdir, pytestconfig, request,
):

    output_dir = run_script(
        script_file=str(pytestconfig.rootpath / 'scripts' / 'singlecam_example.py'),
        input_dir=str(pytestconfig.rootpath / 'data/ibl-pupil'),
        output_dir=tmpdir,
        s=10,
    )
    compare_to_golden(request.node.name, output_dir)
