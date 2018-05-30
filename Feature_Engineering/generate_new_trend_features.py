import pandas as pd
import pickleFileIO
def compute_judgeGrant_dict(df_):
    '''
    This returns a dictionary that stores the information of the grant number and total case number 
    of the given composite keys, ij_code_index and year_month.

    return : dictionary{key: ij_code_index, value: {key: year_month: value: [grant, total]}}
    '''
    judgeGrant_dict = {}
    years = range(df_['adj_year'].min(), df_['adj_year'].max()+1)
    months = range(1, 13)
    check_sum = 0
    for judge_id in df_['ij_code_index'].unique():
        month_to_grant = {}
        for year in years:
            for month in months:
                grantInfo = {}
                grantList = df_[(df_['ij_code_index'] == judge_id) & (df_["adj_year"] == year) 
                                & (df_["adj_month"] == month)]['grant']
                grant=0 
                deny = 0
                if len(grantList) != 0:
                    for g in grantList:
                        if g == 0:
                            deny+=1
                        else:
                            grant+=1
                values = [grant, grant + deny]
                month_to_grant[str(year) + '_' + str(month)] = values
                check_sum += grant + deny
#                 print(str(judge_id) + ' ' + str(year) + '_' + str(month) + ' : ' + str(count))
        judgeGrant_dict[judge_id] = month_to_grant
    print('judgeGrant_dict check_sum = ' + str(check_sum))
    return judgeGrant_dict 
def compute_courtGrant_dict(df_):
    '''
    This returns a dictionary that stores the information of the grant number and total case number 
    of the given composite keys, ij_court_code and year_month.
    
    return : dictionary{key: ij_court_code, value: {key: year_month: value: [grant, total]}}
    '''
    courtGrant_dict = {}
    years = range(df_['adj_year'].min(), df_['adj_year'].max()+1)
    months = range(1, 13)
    check_sum = 0
    for court_id in df_['ij_court_code'].unique():
        month_to_grant = {}
        for year in years:
            for month in months:
                grantInfo = {}
                grantList = df_[(df_['ij_court_code'] == court_id) & (df_["adj_year"] == year) 
                                & (df_["adj_month"] == month)]['grant']
                grant=0 
                deny = 0
                if len(grantList) != 0:
                    for g in grantList:
                        if g == 0:
                            deny+=1
                        else:
                            grant+=1
                values = [grant, grant + deny]
                month_to_grant[str(year) + '_' + str(month)] = values
                check_sum += grant + deny
        courtGrant_dict[court_id] = month_to_grant
    print('courtGrant_dict check_sum = ' + str(check_sum))
    return courtGrant_dict

def add_judge_court_grantRatio_prevMonth(df_):
    '''
    This rgenerate two new trend features using court data.
    ['judge_grantRatio_prevMonth']: #of grant/# of total cases by judge in previous month of adj_date 
    ['court_grantRatio_prevMonth']: #of grant/# of total cases by court in previous month of adj_date 

    df_ : input court dataframe contains court case features
    return : df_ addeded two features
    '''
    df_['judge_grantRatio_prevMonth'] = 0
    df_['court_grantRatio_prevMonth'] = 0
    judgeGrant_dict = compute_judgeGrant_dict(df_)
    courtGrant_dict = compute_courtGrant_dict(df_)
    for index, row in df_.iterrows():
        judge = df_.loc[index, 'ij_code_index']
        y = df_.loc[index, "adj_year"]
        m = df_.loc[index, "adj_month"]
        m = m-1
        if m <= 0:
            m = m + 12
            y = y - 1
        values = [0,0]
        if judge in judgeGrant_dict.keys():
            if str(y)+'_'+str(m) in judgeGrant_dict[judge]:
                #values stores [granted_cases, cases] 
                values = judgeGrant_dict[judge][str(y)+'_'+str(m)]
        ratio = 0
        if values[1]!=0:
            ratio = values[0] / values[1]
        df_.loc[index, 'judge_grantRatio_prevMonth'] = ratio
        ################################################################
        court = df_.loc[index, 'ij_court_code']
        values = [0,0]
        if court in courtGrant_dict.keys():
            if str(y)+'_'+str(m) in courtGrant_dict[court]:
                values = courtGrant_dict[court][str(y)+'_'+str(m)]
        ratio = 0
        if values[1]!=0:
            ratio = values[0] / values[1]
        df_.loc[index, 'court_grantRatio_prevMonth'] = ratio
    return df_


def main():
    """Add two additional feature."""
    court_path = '/data/WorkData/media_and_judging/data/prepared/court_judge_case_trend_wthr_Data_inner_sorted.csv'
    df = pd.read_csv(court_path)
    print(df.shape)
    df = add_judge_court_grantRatio_prevMonth(df)
    df = df.sort_values(['adj_year', 'adj_month', 'adj_day'], ascending=[1, 1, 1])
    df.to_csv('/data/WorkData/media_and_judging/data/prepared/court_judge_case_trend_wthr_Data_inner_reDefTrend.csv')
    pickleFileIO.pickle_dump(df, '/data/WorkData/media_and_judging/data/prepared/court_judge_case_trend_wthr_Data_inner_reDefTrend.p')
if __name__ == '__main__':
    main()
