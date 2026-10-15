from mindscape_delivery.native_demo import native_status

def test_native_method_and_limits_are_explicit():
 status=native_status()
 assert status['optimizer_updates_this_demo']==0
 assert status['cbse_natural_language_supported'] is False
 assert status['trained_parameter_count']==54014
 assert 'masked outside physics' in status['architecture']['scope']
 assert status['historical_results']['coding_iid_exact_percent']==0
