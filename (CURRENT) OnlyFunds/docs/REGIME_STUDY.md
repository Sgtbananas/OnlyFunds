# Regime detector study (Binance, 2019-01-01 -> 2024-12-31; TEST window excluded)

Detector: `core/regime.py` (causal, fixed thresholds, may answer "uncertain").
Reference: ex-post forward trend over {'1h': 48, '4h': 12, '1d': 5} bars, in volatility units (**non-causal, evaluation only**).
Agreement is measured on direction (up / range / down):
- `agreement_with_trailing_direction`: vs the PAST [48, 12, 5] bars (causal), i.e. does it describe the present?
- `agreement_with_expost_direction`: vs the NEXT bars (non-causal), i.e. does the current regime persist / predict?
- `chance_agreement`: what independent labels with the same mix would score against the ex-post label.

## Averages over 9 coins
```
interval  persistence  agreement_with_trailing_direction  agreement_with_expost_direction  chance_agreement  transitions_per_1000_bars  median_run_bars  false_transition_share  median_detection_delay_bars  uncertain_share
      1h            1                           0.577416                         0.292172          0.285147                 146.219334         3.333333                0.438397                    11.555556         0.146249
      1h            4                           0.559481                         0.289670          0.286270                  59.880037        11.888889                0.060360                    12.777778         0.146249
      4h            1                           0.424827                         0.293416          0.288713                 142.478463         3.444444                0.431016                     9.944444         0.146058
      4h            4                           0.369238                         0.293042          0.290577                  59.098401        11.777778                0.064463                    11.833333         0.146058
```

## Per coin
```
interval  symbol  persistence  bars  transitions_per_1000_bars  agreement_with_trailing_direction  median_run_bars  false_transition_share  agreement_with_expost_direction  chance_agreement  median_detection_delay_bars  uncertain_share  share_strong_down  share_weak_down  share_range  share_weak_up  share_strong_up
      1h BTCUSDT            1 52479                 142.761867                           0.582938              3.0                0.435531                         0.286605          0.284844                         13.0         0.145810           0.202424         0.162656     0.200442       0.163265         0.271213
      1h BTCUSDT            4 52479                  57.184779                           0.560681             12.0                0.052982                         0.282600          0.286040                         15.0         0.145810           0.209055         0.152423     0.203396       0.155167         0.279960
      1h ETHUSDT            1 52479                 138.055222                           0.596639              4.0                0.420428                         0.296599          0.288497                         12.0         0.141794           0.201604         0.165514     0.207359       0.159187         0.266335
      1h ETHUSDT            4 52479                  58.023209                           0.579641             12.0                0.056158                         0.293414          0.290380                         12.0         0.141794           0.204634         0.160369     0.212371       0.145849         0.276777
      1h BNBUSDT            1 52479                 144.896054                           0.582709              3.0                0.440163                         0.301024          0.286174                         11.0         0.144230           0.193582         0.167343     0.208541       0.168829         0.261705
      1h BNBUSDT            4 52479                  59.661960                           0.565064             12.0                0.058767                         0.299784          0.287814                         12.0         0.144230           0.199299         0.159130     0.212656       0.157835         0.271080
      1h XRPUSDT            1 52479                 151.889327                           0.555879              3.0                0.450884                         0.291965          0.282670                         11.0         0.150986           0.211780         0.193906     0.217420       0.176814         0.200080
      1h XRPUSDT            4 52479                  61.281656                           0.540788             12.0                0.065609                         0.290324          0.284115                         13.0         0.150986           0.217477         0.187389     0.220679       0.168620         0.205835
      1h LTCUSDT            1 52479                 148.211666                           0.586940              3.0                0.444973                         0.285099          0.285521                         13.0         0.146076           0.209741         0.167515     0.210427       0.176185         0.236133
      1h LTCUSDT            4 52479                  60.043065                           0.567046             12.0                0.063789                         0.280769          0.284913                         14.0         0.146076           0.216925         0.162103     0.208864       0.169115         0.242992
      1h ADAUSDT            1 52479                 147.544732                           0.582252              3.0                0.441689                         0.296409          0.284533                         11.0         0.147522           0.227386         0.172278     0.207969       0.169115         0.223251
      1h ADAUSDT            4 52479                  59.585739                           0.563806             12.0                0.059162                         0.292441          0.285725                         12.0         0.147522           0.236742         0.162103     0.211075       0.157149         0.232931
      1h EOSUSDT            1 52479                 147.316069                           0.566284              4.0                0.437201                         0.288474          0.283206                         10.0         0.146970           0.212561         0.178700     0.208560       0.173708         0.226472
      1h EOSUSDT            4 52479                  60.786219                           0.546733             12.0                0.063636                         0.285671          0.283219                         11.0         0.146970           0.219059         0.169172     0.208522       0.166829         0.236418
      1h XLMUSDT            1 52479                 149.869472                           0.559576              4.0                0.438907                         0.288570          0.282243                         11.0         0.149806           0.223118         0.192934     0.212009       0.164999         0.206940
      1h XLMUSDT            4 52479                  62.501191                           0.544427             11.0                0.059146                         0.288894          0.283549                         13.0         0.149806           0.230511         0.184798     0.215057       0.155738         0.213895
      1h TRXUSDT            1 52479                 145.429600                           0.583529              3.0                0.435797                         0.294807          0.288634                         12.0         0.143050           0.197565         0.158349     0.211246       0.175918         0.256922
      1h TRXUSDT            4 52479                  59.852512                           0.567141             12.0                0.063992                         0.293128          0.290670                         13.0         0.143050           0.205130         0.148174     0.216658       0.164599         0.265440
      4h BTCUSDT            1 13078                 133.200795                           0.434164              4.0                0.413318                         0.291750          0.288178                         10.0         0.141097           0.198043         0.161187     0.195749       0.134883         0.310139
      4h BTCUSDT            4 13078                  57.195290                           0.378881             12.0                0.053476                         0.291597          0.288693                         12.0         0.141097           0.209130         0.151323     0.197507       0.119972         0.322068
      4h ETHUSDT            1 13078                 131.595045                           0.443187              4.0                0.395700                         0.291061          0.292929                         11.0         0.131589           0.205689         0.155681     0.207218       0.129225         0.302187
      4h ETHUSDT            4 13078                  56.736504                           0.389203             11.0                0.059299                         0.289989          0.296474                         13.0         0.131589           0.213106         0.142453     0.217464       0.112555         0.314421
      4h BNBUSDT            1 13078                 141.458939                           0.422465              3.0                0.439459                         0.305755          0.287609                          9.0         0.146878           0.183744         0.173803     0.195137       0.156217         0.291100
      4h BNBUSDT            4 13078                  59.030433                           0.368787             11.0                0.072539                         0.302924          0.290704                         12.0         0.146878           0.191314         0.163939     0.204236       0.139318         0.301193
      4h XRPUSDT            1 13078                 149.793546                           0.416501              3.0                0.444104                         0.292515          0.286543                         11.0         0.153571           0.243845         0.187567     0.212035       0.157364         0.199189
      4h XRPUSDT            4 13078                  61.018504                           0.363588             12.0                0.071429                         0.294658          0.290231                         13.0         0.153571           0.255697         0.172045     0.221594       0.150023         0.200642
      4h LTCUSDT            1 13078                 150.481725                           0.428353              3.0                0.439024                         0.275448          0.291375                         11.0         0.148247           0.203854         0.192690     0.213488       0.143523         0.246444
      4h LTCUSDT            4 13078                  60.865576                           0.361447             12.0                0.061558                         0.273841          0.291472                         12.5         0.148247           0.214176         0.185579     0.213794       0.127160         0.259290
      4h ADAUSDT            1 13078                 143.370546                           0.429347              3.0                0.428800                         0.300245          0.287907                          9.0         0.144748           0.249579         0.163098     0.203395       0.147117         0.236810
      4h ADAUSDT            4 13078                  59.642147                           0.379492             11.0                0.066667                         0.304454          0.291457                         10.0         0.144748           0.263878         0.145894     0.213565       0.130219         0.246444
      4h EOSUSDT            1 13078                 139.012081                           0.412601              4.0                0.424092                         0.288688          0.288295                         10.0         0.140564           0.251491         0.162181     0.208518       0.158510         0.219300
      4h EOSUSDT            4 13078                  59.642147                           0.353877             12.0                0.065385                         0.284479          0.289164                         12.0         0.140564           0.255544         0.154764     0.210889       0.150405         0.228399
      4h XLMUSDT            1 13078                 154.152011                           0.411378              3.0                0.476190                         0.293816          0.283560                          8.5         0.161330           0.272060         0.186802     0.197354       0.134807         0.208977
      4h XLMUSDT            4 13078                  59.030433                           0.362211             12.0                0.075130                         0.300551          0.285492                         10.0         0.161330           0.281771         0.178009     0.202401       0.122419         0.215400
      4h TRXUSDT            1 13078                 139.241474                           0.425447              4.0                0.418451                         0.301469          0.292018                         10.0         0.146497           0.177397         0.168833     0.208136       0.184202         0.261431
      4h TRXUSDT            4 13078                  58.724576                           0.365652             13.0                0.054688                         0.294887          0.291509                         12.0         0.146497           0.185044         0.162640     0.206759       0.173956         0.271601
```

## Findings
1. **Describes the present moderately well.**
   - At 1h it agrees with the trailing 48-bar direction 56–58% of the time (well above the ~33% of random three-way labels).
   - At 4h the agreement is lower (37–42%): the trend score's windows are short relative to a 48-day look-back.
2. **Does not predict the future at all.**
   - Agreement with the next-48-bar direction is 29%, the same as chance (28.5–29%), at every interval and setting.
   - The current regime label has no measurable persistence into the next 2 days (1h) or 8 days (4h).
   - This is consistent with the failure of the regime-driven `auto` strategy in the real-market evaluation.
3. **Unstable without hysteresis.**
   - Raw labels change 146 times per 1,000 bars, with a median run of 3.3 bars, and 44% of changes revert within 6 bars.
   - With 4-bar persistence: 60 changes per 1,000 bars, median run about 12 bars, 6% reverting. The cost is about 1 bar of extra delay (median detection delay rises from about 11.6 to 12.8 bars).
4. **About 15% of bars are labelled "uncertain".**

**Use in OF2:** the detector is used for **monitoring and as a no-trade condition** (an uncertain regime means no trade).
It is **not** evidence for choosing a strategy. Any regime-conditional strategy must establish its own out-of-sample
evidence per regime under the v2 protocol.
