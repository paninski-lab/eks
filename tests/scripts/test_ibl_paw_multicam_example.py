

def test_ibl_paw_multicam_example_defaults(
    run_script, compare_to_golden, tmpdir, pytestconfig, request,
):

    output_dir = run_script(
        script_file=str(pytestconfig.rootpath / 'scripts' / 'ibl_paw_multiview_example.py'),
        input_dir=str(pytestconfig.rootpath / 'data' / 'ibl-paw'),
        output_dir=tmpdir,
    )
    compare_to_golden(request.node.name, output_dir)


def test_ibl_paw_multicam_example_fixed_smooth_param(
    run_script, compare_to_golden, tmpdir, pytestconfig, request,
):

    output_dir = run_script(
        script_file=str(pytestconfig.rootpath / 'scripts' / 'ibl_paw_multiview_example.py'),
        input_dir=str(pytestconfig.rootpath / 'data' / 'ibl-paw'),
        output_dir=tmpdir,
        s=10,
    )
    compare_to_golden(request.node.name, output_dir)
