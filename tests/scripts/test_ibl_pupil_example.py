

def test_ibl_pupil_example_defaults(run_script, tmpdir, pytestconfig):

    run_script(
        script_file=str(pytestconfig.rootpath / 'scripts' / 'ibl_pupil_example.py'),
        input_dir=str(pytestconfig.rootpath / 'data' / 'ibl-pupil'),
        output_dir=tmpdir,
    )


def test_ibl_pupil_example_fixed_smooth_param(run_script, tmpdir, pytestconfig):

    run_script(
        script_file=str(pytestconfig.rootpath / 'scripts' / 'ibl_pupil_example.py'),
        input_dir=str(pytestconfig.rootpath / 'data' / 'ibl-pupil'),
        output_dir=tmpdir,
        diameter_s=0.99,
        com_s=0.99,
    )
