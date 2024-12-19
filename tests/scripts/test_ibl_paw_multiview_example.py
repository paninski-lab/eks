

def test_ibl_paw_multiview_example_defaults(run_script, tmpdir, pytestconfig):

    run_script(
        script_file=str(pytestconfig.rootpath / 'scripts' / 'ibl_paw_multiview_example.py'),
        input_dir=str(pytestconfig.rootpath / 'data' / 'ibl-paw'),
        output_dir=tmpdir,
    )
