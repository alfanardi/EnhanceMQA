from flask import Flask, jsonify,request
from dna_db import read_db
import requests
import json
import time
import logging


app = Flask(__name__)
alarm_api = 'https://ineom.telkomsel.co.id/alarmapi/poi_alarms.json?key=59aFPjfxHGpjXNfnwgCgKm39AZdd2MmfbnKtdgqBrUt6k6vNrULp9GLwf4nvhEWf75wXfafk2SD6fbrtwNZy4BJfXCt5ZSCtDmxmSNuZvnquugupFbQVFTEFPHyLueuL&api_version=1&site_id={}&alarm_groups=[1,2,6,9,10,11,12,13,14,15,16,19,23,24,25,26,27]'
q_laccima_table = """
           SELECT table_name 
           FROM information_schema.tables 
           WHERE table_schema='master_lookup'
           AND table_type='BASE TABLE'
           and table_name ~ '^laccima_'
           order by table_name desc limit 1
       """

q_lacci_start = """ 
           select case when enodeb_id is not null then enodeb_id else rad_lac end as lac_start, rad_ci as ci_start 
           from dna_data.dna_ticket 
           where mobile_ticket = '{}'
    """

q_lacci_end = """ 
               select case when enodeb_id_end is not null then enodeb_id_end else rad_lac_end end as lac_end, rad_ci_end as ci_end 
               from dna_data.dna_ticket 
               where mobile_ticket = '{}'
    """

q_siteid = """
    select site_id from master_lookup.{} where lac = {} and cell_id = {} 
"""

q_update_alarm = """
    update dna_data.dna_ticket_test set alarm = '{}' where mobile_ticket = '{}'
"""

def normalize_digit(x):
    print(x)
    return x if x and x.isdigit() else None

@app.route('/api/v1.0/mobile_ticket/<mobile_ticket>')
def search_by_dna_number(mobile_ticket):
    status = 'FAILED'
    try:


        laccima_table = read_db('ams', q_laccima_table)

        lacci_start = (read_db('dna_prod', q_lacci_start.format(mobile_ticket)))
        siteid_start = []
        if not any(i in lacci_start[0] for i in [0, None]):
            siteid_start = read_db('ams', q_siteid.format(laccima_table[0][0],lacci_start[0][0], lacci_start[0][1]))

        lacci_end = (read_db('dna_prod', q_lacci_end.format(mobile_ticket)))
        siteid_end = []
        if not any(i in lacci_end[0] for i in [0, None]):
            siteid_end = read_db('ams', q_siteid.format(laccima_table[0][0],lacci_end[0][0], lacci_end[0][1]))

        alarm_json = []
        if len(siteid_start) > 0:
            response = requests.get(alarm_api.format(siteid_start[0][0]), verify=False)
            print(response.text)
            data = json.loads(response.text)
            print(siteid_start[0][0])
            if len(data['result']) > 0:
                alarm_json.extend(data['result'])

        if len(siteid_end) > 0 and (siteid_start != siteid_end):
            time.sleep(3)
            response2 = requests.get(alarm_api.format(siteid_end[0][0]), verify=False)
            print(response2.text)
            data2 = json.loads(response2.text)
            if len(data2['result']) > 0:
                alarm_json.extend(data2['result'])
        for i in alarm_json:
            print(i)
        if len(alarm_json) > 0:
            alarm = json.dumps({'active_alarms': alarm_json })
            read_db('dna_prod', q_update_alarm.format(alarm, mobile_ticket), 'update')

        status = 'SUCCESS'
    except Exception as e:
        print(e)

    return jsonify({'status': status})


@app.route('/api/v1.0/get_alarm')
def search_alarm():
    alarm =  ''
    try:
        query_parameters = request.args

        lac_start = normalize_digit(query_parameters.get('lac_start'))
        lac_end = normalize_digit(query_parameters.get('lac_end'))

        enodeb_start = normalize_digit(query_parameters.get('enodeb_start'))
        enodeb_end = normalize_digit(query_parameters.get('enodeb_end'))

        ci_start = normalize_digit(query_parameters.get('ci_start'))
        ci_end = normalize_digit(query_parameters.get('ci_end'))


        lac_start = enodeb_start if (enodeb_start not in [0, None]) else lac_start
        lac_end = enodeb_end if enodeb_end not in [0, None] else lac_end

        laccima_table = read_db('ams', q_laccima_table)

        siteid_start = []
        if not any(i in [lac_start, ci_start] for i in [0, None]):
            siteid_start = read_db('ams', q_siteid.format(laccima_table[0][0], lac_start, ci_start))
        siteid_end = []
        if not any(i in [lac_end, ci_end] for i in [0, None]):
            siteid_end = read_db('ams', q_siteid.format(laccima_table[0][0], lac_end, ci_end))

        if (len(siteid_end) == 0) and (len(siteid_start) == 0):
            alarm = json.dumps({'status': 'SITE NOT FOUND', 'active_alarms': []})
        else:
            alarm_json = []
            if len(siteid_start) > 0:
                print(siteid_start[0][0])
                response = requests.get(alarm_api.format(siteid_start[0][0]), verify=False)
                data = json.loads(response.text)
                if len(data['result']) > 0:
                    alarm_json.extend(data['result'])

            if len(siteid_end) > 0 and (siteid_start != siteid_end):
                time.sleep(3)
                print(siteid_end[0][0])
                response2 = requests.get(alarm_api.format(siteid_end[0][0]), verify=False)
                data2 = json.loads(response2.text)
                if len(data2['result']) > 0:
                    alarm_json.extend(data2['result'])
            alarm = json.dumps({'status': 'SUCCESS', 'active_alarms': alarm_json})
    except Exception as e:
        alarm = json.dumps({'status': 'ERROR', 'active_alarms': []})
        logging.info(e)
        print(e)
    return alarm

if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    app.run(host = '0.0.0.0', port=7001)
