"""
This python script holds:
1. PATH to data
2. Feature Selection
"""

"""
PATHs
"""
# PICKLE PATHs
PICKLE_PATH = '../data/pickles/'
# ROOT PATHs
COURT_PATH = '../../data_ignore/'
NEWS_PATH = '../data/news/old_news/'
CABLES_PATH = '../data/calbes/'

# COURT
COURT_CASE_FOR_PROPROCESS_PATH = COURT_PATH + '_decision_scheduling_merge_final_converted.csv' #for feature generation
COURT_CASE_PATH = '../../trainingdata/court_judge_case_trend_wthr_Data_inner.csv' #for traning
DOC_PATH = '../../DocumentEmbedding/news8710/newsobject8710.p'
W2V_PATH = '../../newsMetaData/news0206.vec'
EMBED_PATH = '../../newsMetaData/title2embedding.p'

# OPTION
GROUPING = True
"""
FEATURES
"""

# FULL JUDGE BIOS FEATURES
# JUDGE BIOS TO KEEP

judge_features_to_keep = ['Male_judge', 'Year_Appointed_SLR',
                          'Year_College_SLR', 'Year_Law_school_SLR',
                          'Government_Years_SLR', 'Govt_nonINS_SLR',
                          'INS_Years_SLR', 'Military_Years_SLR',
                          'NGO_Years_SLR', 'Privateprac_Years_SLR',
                          'Academia_Years_SLR', 'ij_code']

# FULL COURT CASE FEATURES

# COURT CASE FEATURES TO KEEP
courtcase_features_to_keep = ['ij_code', 'tracid', 'comp_year', 'comp_date',
                              'appl_dec', 'appl_code', 'nat', 'case_type',
                              'c_asy_type', 'base_city_code',
                              'hearing_loc_code', 'hearing_loc_city',
                              'hearing_loc_state', 'base_city_state',
                              'attorney_flag', 'osc_year', 'osc_month',
                              'osc_day', 'osc_date', 'adj_date_stamp',
                              'adj_time_start', 'schedule_type', 'langid',
                              'input_year', 'input_month', 'input_day',
                              'appl_year', 'appl_month', 'appl_day',
                              'defensive', 'affirmative', 'generation',
                              'hearingid', 'adj_year', 'adj_month', 'adj_day',
                              'grant', 'deny', 'CaseId']


news_features = ['newsgroup' + str(i) for i in range(1, 301)]
"""
Miscellaneous
"""

# NATIONALITY CODES
country_codes = {
   2:'AB',3:'AC',4:'AF',5:'AG',6:'AL',7:'AM',8:'AN',9:'AO',10:'AR',11:'AS',12:'AU',13:'AV',14:'AZ',
   15:'BA',16:'BB',17:'BC',18:'BD',19:'BE',20:'BF',21:'BG',22:'BH',23:'BI',24:'BL',25:'BM',26:'BN',
   27:'BO',28:'BP',29:'BR',30:'BS',31:'BT',32:'BU',33:'BV',34:'BW',35:'BX',36:'BY',37:'BZ',38:'CA',
   39:'CB',40:'CC',41:'CD',42:'CE',43:'CF',44:'CG',45:'CH',46:'CI',47:'CJ',48:'CK',49:'CM',50:'CN',
   51:'CO',52:'CR',53:'CS',54:'CT',55:'CU',56:'CV',57:'CW',58:'CX',59:'CY',60:'CZ',61:'DA',62:'DC',
   63:'DJ',64:'DM',65:'DO',66:'DR',67:'EC',68:'EG',69:'EI',70:'EK',71:'EO',72:'ER',73:'ES',74:'ET',
   75:'FA',76:'FG',77:'FI',78:'FJ',79:'FM',80:'FO',81:'FP',82:'FR',83:'FS',84:'FW',85:'GA',86:'GB',
   87:'GC',88:'GE',89:'GH',90:'GI',91:'GJ',92:'GL',93:'GO',94:'GP',95:'GR',96:'GT',97:'GV',98:'GY',
   99:'GZ',100:'HA',101:'HK',102:'HL',103:'HM',104:'HO',105:'HU',106:'IC',107:'ID',108:'IN',109:'IO',
   110:'IR',111:'IS',112:'IT',113:'IV',114:'IZ',115:'JA',116:'JM',117:'JO',118:'KE',119:'KG',120:'KN',
   121:'KR',122:'KS',123:'KT',124:'KU',125:'KV',126:'KZ',127:'LA',128:'LE',129:'LH',130:'LI',131:'LS',
   132:'LT',133:'LU',134:'LV',135:'LY',136:'MA',137:'MB',138:'MC',139:'MD',140:'MG',141:'MH',142:'MI',
   247:'MJ',143:'ML',144:'MM',145:'MN',146:'MO',147:'MP',148:'MQ',149:'MR',150:'MT',151:'MU',152:'MV',
   153:'MX',154:'MY',155:'MZ',156:'NA',157:'NC',158:'NE',159:'NF',160:'NG',161:'NH',162:'NI',163:'NL',
   164:'NN',165:'NO',166:'NP',167:'NR',168:'NS',169:'NU',170:'NZ',171:'PA',172:'PC',173:'PE',174:'PK',
   175:'PL',176:'PM',177:'PN',178:'PO',179:'PP',180:'PS',181:'PU',182:'QA',183:'RE',184:'RM',185:'RO',
   186:'RP',187:'RU',188:'RW',189:'SA',190:'SB',191:'SC',192:'SE',193:'SF',194:'SG',195:'SH',196:'SK',
   197:'SL',198:'SM',199:'SN',200:'SO',201:'SP',202:'SR',203:'SS',204:'ST',205:'SU',206:'SV',207:'SW',
   208:'SY',209:'SZ',210:'TA',211:'TC',212:'TD',213:'TH',214:'TK',215:'TL',216:'TM',217:'TN',218:'TO',
   219:'TP',220:'TR',221:'TS',222:'TU',223:'TV',224:'TW',225:'TZ',226:'UE',227:'UG',228:'UK',229:'UR',
   230:'UV',231:'UY',232:'UZ',233:'VC',234:'VE',235:'VI',236:'VM',237:'WA',238:'WI',239:'WS',240:'WZ',
   251:'XS',241:'XX',242:'YE',243:'YO',244:'YS',245:'ZA',246:'ZI', 1000: 'Unknown', 1: 'Unknown'
}
