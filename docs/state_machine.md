# State Machine (schelet)

IDLE -> ACQUIRE_DATA -> PREPROCESS -> INFERENCE -> DISPLAY/ACT -> LOG -> [ERROR] -> STOP

Justificare:
- Am ales un flux de tip monitorizare în timp real: date sunt colectate, preprocesate, trecute prin RN și rezultatul afișat/înregistrat.
- State ERROR gestionează pierderi de date, timeouts sau valori abnormally high.

Stări principale:
1. IDLE: așteaptă trigger (start manual sau cron)
2. ACQUIRE_DATA: citește batch de la `src/data_acquisition` sau din CSV
3. PREPROCESS: aplică scaler și filtre (src/preprocessing)
4. INFERENCE: modelul antrenat realizează predicția
5. DISPLAY/ACT: UI arată rezultatul sau trimite o acțiune
6. LOG: salvează predicția și metadate
7. ERROR: retry/recover or safe shutdown
