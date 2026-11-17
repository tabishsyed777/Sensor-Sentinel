"""Causal time-series anomaly detection for one regularly sampled machine."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score

SENSORS = ['temperature', 'vibration', 'pressure', 'load']
LOOKBACK = 12
MINUTES = 5


def generate(n=7200, seed=42):
    if n < 1200: raise ValueError('At least 1200 readings required')
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    load = 55 + 15*np.sin(t*2*np.pi/288) + rng.normal(0,2,n)
    frame = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01',periods=n,freq='5min',tz='UTC'),
        'temperature': 30+.45*load+rng.normal(0,.8,n),
        'vibration': .7+.025*load+rng.normal(0,.12,n),
        'pressure': 4.9-.007*load+rng.normal(0,.05,n),
        'load': load, 'anomaly': np.zeros(n,dtype=int)})
    # Healthy training half; isolated episodes in calibration and test quarters.
    for begin,end,count in [(n//2+LOOKBACK,n*3//4,4),(n*3//4+LOOKBACK,n,5)]:
        positions = np.linspace(begin+30,end-60,count).astype(int)
        for i,start in enumerate(positions):
            length = 18 + i*3
            idx = np.arange(start,start+length)
            sensor = SENSORS[i%3]
            magnitude = {'temperature':12.,'vibration':1.8,'pressure':-.8}[sensor]
            # Include both sudden and gradually emerging faults.
            ramp = np.linspace(.2,1,length) if i%2 else np.ones(length)
            frame.loc[idx,sensor] += magnitude*ramp
            frame.loc[idx,'anomaly'] = 1
    return frame


def validate(frame):
    if set(frame.columns) != set(['timestamp', 'anomaly']+SENSORS):
        raise ValueError('Expected exact documented schema; unknown columns may leak outcomes')
    data = frame.copy().reset_index(drop=True)
    if len(data)<600: raise ValueError('At least 600 readings required')
    data['timestamp'] = pd.to_datetime(data.timestamp,utc=True,errors='raise')
    if data.timestamp.isna().any() or not data.timestamp.diff().iloc[1:].eq(pd.Timedelta(minutes=MINUTES)).all():
        raise ValueError('Timestamps must increase at a regular five-minute cadence, without duplicates or gaps')
    for col in SENSORS:
        data[col] = pd.to_numeric(data[col],errors='raise')
    if not np.isfinite(data[SENSORS].to_numpy(dtype=float)).all():
        raise ValueError('Sensor readings must be finite and nonmissing')
    if data.anomaly.isna().any() or not data.anomaly.isin([0,1]).all():
        raise ValueError('Binary anomaly labels required for offline evaluation')
    return data


def features(data):
    values = data[SENSORS].astype(float)
    result = values.copy()
    # Exclude the current reading from historical windows; never use centered windows or backfill.
    history = values.shift(1).rolling(LOOKBACK,min_periods=LOOKBACK)
    for col in SENSORS:
        result[col+'_past_mean'] = history.mean()[col]
        result[col+'_past_std'] = history.std(ddof=0)[col]
        result[col+'_delta'] = values[col]-history.mean()[col]
    return result.iloc[LOOKBACK:]


def split(data):
    x = features(data)
    first, second = len(data)//2, len(data)*3//4
    # Purge LOOKBACK readings after boundaries so feature windows cannot overlap splits.
    return (x[x.index<first],
            x[(x.index>=first+LOOKBACK)&(x.index<second)],
            x[x.index>=second+LOOKBACK])


def calibrate(normal_scores, budget=.01):
    scores = np.asarray(normal_scores,dtype=float)
    if not 0<budget<1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Finite normal calibration scores and a budget between zero and one required')
    # Strict > comparison preserves a conservative empirical budget in the presence of ties.
    return float(np.quantile(scores,1-budget,method='higher'))


def starts(mask):
    mask = np.asarray(mask,dtype=bool)
    return np.flatnonzero(mask & ~np.r_[False,mask[:-1]])


def event_metrics(y, flagged):
    y,flagged = np.asarray(y,dtype=bool),np.asarray(flagged,dtype=bool)
    beginnings = starts(y)
    endings = np.flatnonzero(y & ~np.r_[y[1:],False])+1
    delays=[]
    for begin,end in zip(beginnings,endings):
        alarms=np.flatnonzero(flagged[begin:end])
        if len(alarms): delays.append(int(alarms[0]*MINUTES))
    normal_count=int((~y).sum())
    false=flagged & ~y
    normal_days=normal_count*MINUTES/(60*24)
    return {'events':len(beginnings),'detected_events':len(delays),'missed_events':len(beginnings)-len(delays),
            'event_recall':len(delays)/len(beginnings) if len(beginnings) else None,
            'detected_event_delays_minutes':delays,
            'median_detected_event_delay_minutes':float(np.median(delays)) if delays else None,
            'false_alarm_readings':int(false.sum()),'normal_readings':normal_count,
            'false_positive_rate':float(false.sum()/normal_count) if normal_count else None,
            'false_alarm_episodes':len(starts(false)),
            'false_alarm_episodes_per_normal_day':len(starts(false))/normal_days if normal_days else None}


def summarize(y,scores,threshold):
    flagged=scores>threshold
    return {**event_metrics(y,flagged),
            'average_precision':float(average_precision_score(y,scores)),
            'roc_auc':float(roc_auc_score(y,scores)),
            'point_precision':float(precision_score(y,flagged,zero_division=0)),
            'point_recall':float(recall_score(y,flagged,zero_division=0))}


def run(frame,budget=.01):
    data=validate(frame)
    first,second=len(data)//2,len(data)*3//4
    if data.anomaly.iloc[:first].any():
        raise ValueError('Initial training period must be independently certified healthy')
    train,validation,test=split(data)
    for part in [validation,test]:
        if data.loc[part.index,'anomaly'].nunique()!=2:
            raise ValueError('Calibration and test periods must contain normal and anomaly labels')
    model=IsolationForest(n_estimators=160,max_samples=512,random_state=42,n_jobs=1)
    model.fit(train)
    val_scores=-model.score_samples(validation)
    scores=-model.score_samples(test)
    val_y=data.loc[validation.index,'anomaly'].to_numpy()
    y=data.loc[test.index,'anomaly'].to_numpy()
    threshold=calibrate(val_scores[val_y==0],budget)
    # Simple comparator: maximum absolute robust deviation of raw sensors from healthy training.
    center=train[SENSORS].median()
    scale=((train[SENSORS]-center).abs().median()*1.4826).clip(lower=1e-9)
    baseline_val=((validation[SENSORS]-center)/scale).abs().max(axis=1).to_numpy()
    baseline_test=((test[SENSORS]-center)/scale).abs().max(axis=1).to_numpy()
    baseline_threshold=calibrate(baseline_val[val_y==0],budget)
    periods={name:{'first':str(data.loc[part.index,'timestamp'].min()),
                   'last':str(data.loc[part.index,'timestamp'].max())}
             for name,part in [('train',train),('calibration',validation),('test',test)]}
    report={'data_origin':'Caller-supplied data; CLI default is synthetic',
            'rows':{'train':len(train),'calibration':len(validation),'test':len(test),
                    'warmup_and_purged':len(data)-len(train)-len(validation)-len(test)},
            'periods':periods,'lookback_readings':LOOKBACK,'cadence_minutes':MINUTES,
            'calibration_false_positive_budget':budget,'threshold':threshold,
            'observed_calibration_false_positive_rate':float(np.mean(val_scores[val_y==0]>threshold)),
            'isolation_forest':summarize(y,scores,threshold),
            'robust_baseline':summarize(y,baseline_test,baseline_threshold),
            'robust_baseline_threshold':baseline_threshold,
            'score_meaning':'Higher Isolation Forest score means more unusual, not failure probability.',
            'limitations':['Synthetic faults do not establish real-equipment performance.',
                           'Calibration uses labeled normal readings; future false-alarm budgets are not guaranteed.',
                           'Temporal splitting evaluates one machine over time, not unseen-machine generalization.',
                           'Detected-event delay excludes missed events; inspect missed-event count alongside delay.',
                           'Historical windows may remain elevated after faults, producing recovery-period alerts.']}
    predictions=data.loc[test.index,['timestamp']+SENSORS+['anomaly']].copy()
    predictions['score']=scores
    predictions['flagged']=scores>threshold
    predictions['baseline_score']=baseline_test
    predictions['baseline_flagged']=baseline_test>baseline_threshold
    return report,predictions


def plot(report,predictions,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    times=pd.to_datetime(predictions.timestamp)
    for ax,sensor in zip(axes[:2],['temperature','vibration']):
        ax.plot(times,predictions[sensor],lw=.8,color='#3268ad')
        ax.set_ylabel(sensor)
    axes[2].plot(times,predictions.score,lw=.8,label='Anomaly score',color='#3268ad')
    axes[2].axhline(report['threshold'],color='#c34a36',ls='--',label='Calibration threshold')
    flags=predictions.flagged.to_numpy()
    axes[2].scatter(times[flags],predictions.score[flags],s=9,color='#c34a36',label='Flagged')
    y=predictions.anomaly.to_numpy(dtype=bool)
    ends=np.flatnonzero(y & ~np.r_[y[1:],False])
    for begin,end in zip(starts(y),ends):
        for ax in axes: ax.axvspan(times.iloc[begin],times.iloc[end]+pd.Timedelta(minutes=MINUTES),alpha=.15,color='orange')
    axes[2].set_ylabel('Anomaly score')
    axes[2].legend(loc='upper left')
    fig.suptitle('Sensor Sentinel · held-out timeline · orange shading = labeled fault')
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out/'timeline.png',dpi=150)
    fig.savefig(out/'timeline.svg')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path)
    parser.add_argument('--out',type=Path,default=Path('work/demo'))
    parser.add_argument('--false-positive-budget',type=float,default=.01)
    args=parser.parse_args()
    frame=pd.read_csv(args.csv) if args.csv else generate()
    report,predictions=run(frame,args.false_positive_budget)
    report['data_origin']='User-supplied CSV, provenance unverified' if args.csv else 'Synthetic seed-42 single-machine demo'
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    predictions.to_csv(args.out/'test_predictions.csv',index=False)
    plot(report,predictions,args.out)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
