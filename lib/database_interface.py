#!/usr/bin/python3
import psycopg2

import time
from datetime import datetime
from contextlib import closing


class phys_val_t:
    def __init__(self, v = 0.0, e = 0.0):
        self.value = v
        self.error = e

class pol_val_t:
    def __init__(self, l = phys_val_t(), r = phys_val_t()):
        self.left  = l
        self.right = r


class Db_obj:
    local_id = 0
    begintime = 0
    endtime = 0
    measuretime = 300.
    Eset = -9999.
    Fdep = -9999.
    Edep = -9999.
    Adep = -9999
    Fspeed = -9999
    H    = -9999.

    method = 0
    version = 0
    status  = 0 
    chi2 = -1.
    ndf  =  1

    x = pol_val_t()
    y = pol_val_t()
    asym_x = phys_val_t()
    asym_y = phys_val_t()

    P = phys_val_t()
    Q = phys_val_t()
    V = phys_val_t()
    NL = phys_val_t()
    NR = phys_val_t()
    beta = phys_val_t()
    askewx = phys_val_t()
    askewy = phys_val_t()

    def write(self, parsed_db_args):
        print('Writing to database:'+parsed_db_args['path']+'@'+parsed_db_args['user']+'/'+parsed_db_args['host'])
        with closing(psycopg2.connect(dbname=parsed_db_args['path'], user=parsed_db_args['user'], host=parsed_db_args['host'])) as conn:
            with closing(conn.cursor()) as cursor:
                lst = []
                lst.append( ("begintime", time.strftime("'%Y-%m-%d %H:%M:%S'", time.localtime(self.begintime))))
                lst.append( ("endtime", time.strftime("'%Y-%m-%d %H:%M:%S'", time.localtime(self.endtime)) ))
                lst.append( ("measuretime", "{:4f}".format(self.measuretime)) )
                lst.append( ("eset", "{:.2f}".format(self.Eset)) )
                lst.append( ("fdep", "{:.6f}".format(self.Fdep)) )
                lst.append( ("edep", "{:.6f}".format(self.Edep)) )
                lst.append( ("adep", "{:.1f}".format(self.Adep)) )
                lst.append( ("fspeed", "{:.6f}".format(self.Fspeed)) )
                lst.append( ("h", "{:.6f}".format(self.H)) )

                lst.append( ("xl", "{:.4f}".format(self.x.left.value)) )
                lst.append( ("xr", "{:.4f}".format(self.x.right.value)) )
                lst.append( ("yl", "{:.4f}".format(self.y.left.value)) )
                lst.append( ("yr", "{:.4f}".format(self.y.right.value)) )
                lst.append( ("ax", "{:.4f}".format(self.asym_x.value)) )
                lst.append( ("ay", "{:.4f}".format(self.asym_y.value)) )

                lst.append( ("dxl", "{:4f}".format(self.x.left.error)) )
                lst.append( ("dxr", "{:4f}".format(self.x.right.error)) )
                lst.append( ("dyl", "{:4f}".format(self.y.left.error)) )
                lst.append( ("dyr", "{:4f}".format(self.y.right.error)) )
                lst.append( ("dax", "{:4f}".format(self.asym_x.error)) )
                lst.append( ("day", "{:4f}".format(self.asym_y.error)) )

                lst.append( ("method", "{}".format(self.method)) )
                lst.append( ("version", "{}".format(self.version)) )
                lst.append( ("status", "{}".format(self.status)) )

                lst.append( ("chi2", "{:.6g}".format(self.chi2)) )
                lst.append( ("ndf", "{}".format(self.ndf)) )


                lst.append( ("p", "{:.4f}".format(self.P.value)) )
                lst.append( ("q", "{:.4f}".format(self.Q.value)) )
                lst.append( ("v", "{:.4f}".format(self.V.value)) )
                lst.append( ("nl", "{:.4f}".format(self.NL.value)) )
                lst.append( ("nr", "{:.4f}".format(self.NR.value)) )
                lst.append( ("beta", "{:.4f}".format(self.beta.value)) )

                lst.append( ("dp", "{:.4f}".format(self.P.error)) )
                lst.append( ("dq", "{:.4f}".format(self.Q.error)) )
                lst.append( ("dv", "{:.4f}".format(self.V.error)) )
                lst.append( ("dnl", "{:.4f}".format(self.NL.error)) )
                lst.append( ("dnr", "{:.4f}".format(self.NR.error)) )
                lst.append( ("dbeta", "{:.4f}".format(self.beta.error)) )
                #lst.append( ("askewx", "{:.4f}".format(self.askewx.value)) )
                #lst.append( ("askewy", "{:.4f}".format(self.askewy.value)) )
                #lst.append( ("daskewx", "{:.4f}".format(self.askewx.error)) )
                #lst.append( ("daskewy", "{:.4f}".format(self.askewy.error)) )
                #for l in lst:
                #    print( l[0], "         ", l[1])

                query = "insert into lsrp_pol ("
                for i in range(0, len(lst) ):
                    
                    query += lst[i][0]
                    if i < len(lst) - 1: query +=","
                query += ") VALUES ("
                for i in range(0, len(lst) ):
                    if lst[i][1] != 'nan':
                        query += lst[i][1]
                    else:
                        query += "-9999"
                    if i < len(lst) - 1: query +=","
                query += ");"
                cursor.execute(query)
                conn.commit()
                
def parse_db_args(db_line):
    user_idx = db_line.find('@')
    host_idx = db_line.find('/')
    parsed_db_args = {}
    if user_idx > 0 and host_idx > 0:
        parsed_db_args['user'] = db_line[:user_idx]
        parsed_db_args['host'] = db_line[user_idx+1:host_idx]
        parsed_db_args['path'] = db_line[host_idx+1:]
    else:
        print('Error: invalid database line!')
        print('Check your config file and use the following tebmplate: user@host/path')
    return parsed_db_args
    
def db_write(   db_obj,
                config,
                begin_time,
                end_time,
                fitter,
                env_params,
                e_dep,
                fit_counter, 
                version):
                
    fitres = fitter.minuit
    db_obj.local_id = fit_counter
    db_obj.begintime = begin_time
    db_obj.endtime = end_time+10 #TODO 10 means the time in seconds for the single file. Needs to be written automatically.
    db_obj.measuretime = db_obj.endtime - db_obj.begintime
    db_obj.Eset = env_params['vepp4E']
    db_obj.Fdep = env_params['dfreq']
    db_obj.Adep = env_params['att']
    db_obj.Fspeed = env_params['fspeed']
    db_obj.Edep = e_dep
    db_obj.H = env_params['vepp4H_nmr']
    db_obj.chi2 = fitter.chi2
    db_obj.ndf = fitter.ndf
    db_obj.P.value = fitres.values['P']
    db_obj.P.error = fitres.errors['P']
    db_obj.V.value = fitres.values['V']
    db_obj.V.error = fitres.errors['V']
    db_obj.Q.value = fitres.values['Q']
    db_obj.Q.error = fitres.errors['Q']
    db_obj.NL.value = fitres.values['N']
    db_obj.NR.value = fitres.values['N']
    db_obj.NL.error = fitres.errors['N']
    db_obj.NR.error = fitres.errors['N']
    db_obj.version = version
    db_obj.write(parse_db_args(config['database']))
    #db_obj.write(dbname='calibrations', user='calibrations', host='bison-new')

