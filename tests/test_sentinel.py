import unittest
import numpy as np
from sentinel import generate, validate, features, split, calibrate, event_metrics, run, SENSORS, LOOKBACK

class SentinelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.data=generate(2400)

    def test_seed_reproducible(self):
        self.assertTrue(generate(1200).equals(generate(1200)))

    def test_training_period_healthy(self):
        self.assertEqual(self.data.iloc[:1200].anomaly.sum(),0)

    def test_forbidden_column_rejected(self):
        with self.assertRaises(ValueError): validate(self.data.assign(failure_reason='leaked'))

    def test_nonfinite_reading_rejected(self):
        bad=self.data.copy();bad.loc[10,'temperature']=np.nan
        with self.assertRaises(ValueError): validate(bad)

    def test_irregular_cadence_rejected(self):
        with self.assertRaises(ValueError): validate(self.data.drop(index=10))

    def test_future_readings_do_not_change_past_features(self):
        original=features(self.data)
        altered=self.data.copy();altered.loc[1000:,SENSORS]+=1000
        changed=features(altered)
        np.testing.assert_allclose(original.loc[:999],changed.loc[:999])

    def test_labels_not_features(self):
        changed=self.data.copy();changed['anomaly']=1-changed.anomaly
        np.testing.assert_allclose(features(self.data),features(changed))

    def test_split_windows_do_not_overlap(self):
        train,validation,test=split(self.data)
        self.assertGreater(validation.index.min()-LOOKBACK,train.index.max())
        self.assertGreater(test.index.min()-LOOKBACK,validation.index.max())

    def test_contaminated_training_rejected(self):
        bad=self.data.copy();bad.loc[50,'anomaly']=1
        with self.assertRaisesRegex(ValueError,'healthy'): run(bad)

    def test_threshold_false_positive_budget_with_ties(self):
        normal=np.r_[np.arange(90),np.full(10,100)]
        threshold=calibrate(normal,.05)
        self.assertLessEqual(np.mean(normal>threshold),.05)

    def test_event_detection_delay_and_missed_event(self):
        y=np.array([0,1,1,1,0,1,1,0]);p=np.array([0,0,0,1,0,0,0,0],dtype=bool)
        result=event_metrics(y,p)
        self.assertEqual(result['events'],2)
        self.assertEqual(result['detected_events'],1)
        self.assertEqual(result['detected_event_delays_minutes'],[10])
        self.assertEqual(result['missed_events'],1)

    def test_no_event_is_not_perfect_recall(self):
        self.assertIsNone(event_metrics(np.zeros(10),np.zeros(10,dtype=bool))['event_recall'])

    def test_end_to_end_report(self):
        report,predictions=run(self.data)
        self.assertEqual(len(predictions),report['rows']['test'])
        self.assertIn('robust_baseline',report)
        self.assertTrue(np.isfinite(predictions.score).all())
        self.assertGreater(report['isolation_forest']['events'],0)

    def test_test_labels_do_not_change_scores_or_threshold(self):
        report,a=run(self.data)
        changed=self.data.copy();changed.loc[1800:,'anomaly']=1-changed.loc[1800:,'anomaly']
        second,b=run(changed)
        self.assertEqual(report['threshold'],second['threshold'])
        np.testing.assert_allclose(a.score,b.score)

if __name__=='__main__': unittest.main()
