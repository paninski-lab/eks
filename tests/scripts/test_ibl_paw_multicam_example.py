

def test_ibl_paw_multicam_example_defaults(run_script, tmpdir, pytestconfig):

    run_script(
        script_file=str(pytestconfig.rootpath / 'scripts' / 'ibl_paw_multiview_example.py'),
        input_dir=str(pytestconfig.rootpath / 'data' / 'ibl-paw'),
        output_dir=tmpdir,
    )

def test_ibl_paw_multicam_example_fixed_smooth_param(run_script, tmpdir, pytestconfig):
    run_script(
        script_file=str(pytestconfig.rootpath / 'scripts' / 'ibl_paw_multiview_example.py'),
        input_dir=str(pytestconfig.rootpath / 'data' / 'ibl-paw'),
        output_dir=tmpdir,
        s=10
    )