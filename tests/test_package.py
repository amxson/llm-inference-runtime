def test_package_import() -> None:
    import llm_runtime

    assert llm_runtime.__version__ == "0.1.0"
