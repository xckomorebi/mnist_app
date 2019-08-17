import datetime
import os
from flask import Flask,request
from werkzeug.utils import secure_filename

from preprocess import pre_pic,getPred
import uuid

from cassandra.cluster import Cluster

app = Flask(__name__)

@app.route('/mnist',methods = ["GET","POST"])
def mnist():
    req_time = datetime.datetime.now()

    if request.method == "POST":
        f= request.files['file']
        upload_filename = secure_filename(f.filename)
        save_filename = str(req_time).rsplit('.',1)[0]+''+upload_filename
        save_filepath = os.path.join(app.root_path,'uploads',save_filename)
        f.save(save_filepath)
            
        img = pre_pic(save_filepath)
        pred = str(getPred(img)[0])
        
    
        clusterList=['192.168.0.102','127.0.0.1']
        cluster=Cluster(clusterList)
        session=cluster.connect("bd26_pj")
        session.execute("INSERT INTO mnist(filename,path,time,prediction) VALUES (%s,%s,%s,%s)",[save_filename,save_filepath,req_time,pred])
        
        return("%s%s%s%s%s%s%s%s%s" % ("Filepath: ",save_filepath,'\n',"Filename: ",save_filename,'\n',"Prediction: ",pred,"\n"))

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)
