OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
h q[0];
cx q[0],q[8];
cx q[1],q[8];
h q[2];
cx q[2],q[8];
cx q[3],q[8];
h q[4];
cx q[4],q[8];
cx q[5],q[8];
cx q[6],q[8];
h q[7];
cx q[7],q[8];
rz(-0.015528882) q[8];
cx q[6],q[8];
cx q[2],q[8];
h q[2];
cx q[0],q[8];
h q[7];
cx q[7],q[8];
cx q[5],q[8];
h q[4];
cx q[4],q[8];
cx q[3],q[8];
rz(-0.00070377871) q[8];
cx q[5],q[8];
cx q[3],q[1];
h q[4];
cx q[4],q[1];
h q[7];
cx q[7],q[1];
cx q[8],q[1];
rz(-0.00070377871) q[1];
cx q[8],q[1];
cx q[7],q[1];
h q[7];
cx q[4],q[1];
h q[4];
cx q[3],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
h q[1];
h q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
h q[1];
cx q[3],q[6];
cx q[6],q[3];
cx q[3],q[6];
h q[3];
h q[6];
cx q[3],q[6];
cx q[6],q[3];
cx q[3],q[6];
h q[3];
h q[6];
cx q[4],q[8];
cx q[8],q[4];
cx q[4],q[8];
h q[4];
h q[8];
cx q[4],q[8];
cx q[8],q[4];
cx q[4],q[8];
h q[8];
cx q[5],q[1];
cx q[1],q[5];
cx q[5],q[1];
h q[5];
h q[1];
cx q[5],q[1];
cx q[1],q[5];
cx q[5],q[1];
h q[5];
h q[1];
cx q[7],q[5];
cx q[5],q[7];
cx q[7],q[5];
h q[7];
h q[5];
cx q[7],q[5];
cx q[5],q[7];
cx q[7],q[5];
h q[7];
h q[5];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[5],q[6];
rz(0.0054597657) q[6];
cx q[3],q[6];
cx q[1],q[6];
cx q[5],q[6];
h q[2];
cx q[2],q[6];
h q[0];
cx q[0],q[6];
rz(-0.0042997153) q[6];
cx q[0],q[6];
cx q[2],q[6];
cx q[5],q[6];
cx q[1],q[0];
h q[2];
cx q[2],q[0];
cx q[4],q[0];
cx q[5],q[0];
cx q[6],q[0];
h q[9];
cx q[9],q[0];
rz(-0.016746723) q[0];
cx q[9],q[0];
cx q[6],q[0];
cx q[5],q[0];
cx q[4],q[0];
cx q[2],q[0];
cx q[1],q[0];
h q[9];
cx q[9],q[4];
cx q[6],q[4];
rz(-0.0030813402) q[4];
cx q[6],q[4];
cx q[3],q[8];
rz(0.13116905) q[8];
cx q[3],q[8];
cx q[5],q[8];
rz(0.13960362) q[8];
cx q[5],q[8];
cx q[1],q[0];
h q[2];
cx q[2],q[0];
cx q[5],q[0];
rz(0.0089581491) q[0];
cx q[5],q[0];
cx q[2],q[0];
h q[2];
cx q[1],q[0];
cx q[8],q[0];
rz(0.13464686) q[0];
cx q[8],q[0];
h q[4];
cx q[4],q[9];
cx q[7],q[9];
rz(0.0055514925) q[9];
cx q[7],q[9];
cx q[4],q[9];
h q[4];
cx q[1],q[6];
s q[0];
h q[0];
cx q[0],q[6];
rz(-0.0076365624) q[6];
cx q[0],q[6];
h q[0];
sdg q[0];
cx q[1],q[6];
s q[2];
h q[2];
cx q[2],q[0];
s q[6];
h q[6];
cx q[6],q[0];
rz(0.0055514925) q[0];
cx q[6],q[0];
h q[6];
sdg q[6];
cx q[2],q[0];
cx q[7],q[6];
rz(0.12101121) q[6];
cx q[7],q[6];
cx q[7],q[9];
rz(0.16720244) q[9];
cx q[7],q[9];
cx q[5],q[9];
s q[4];
h q[4];
cx q[4],q[9];
rz(-0.0023645665) q[9];
cx q[4],q[9];
cx q[5],q[9];
h q[1];
cx q[1],q[0];
s q[4];
h q[4];
cx q[4],q[0];
cx q[5],q[0];
s q[8];
h q[8];
cx q[8],q[0];
rz(-0.0069327837) q[0];
cx q[8],q[0];
h q[8];
sdg q[8];
cx q[4],q[0];
cx q[1],q[0];
h q[1];
cx q[7],q[0];
h q[6];
cx q[6],q[0];
cx q[5],q[0];
cx q[1],q[0];
rz(0.021778815) q[0];
cx q[7],q[0];
h q[0];
cx q[0],q[6];
cx q[1],q[6];
cx q[3],q[6];
cx q[5],q[6];
rz(-0.00070377871) q[6];
cx q[5],q[6];
cx q[3],q[6];
cx q[0],q[6];
h q[0];
s q[7];
h q[7];
cx q[7],q[8];
cx q[5],q[8];
s q[4];
h q[4];
cx q[4],q[8];
cx q[3],q[8];
rz(-0.00070377871) q[8];
cx q[3],q[8];
cx q[4],q[8];
cx q[5],q[8];
cx q[7],q[8];
h q[7];
sdg q[7];
cx q[1],q[0];
s q[2];
h q[2];
cx q[2],q[0];
cx q[3],q[0];
cx q[5],q[0];
cx q[6],q[0];
rz(0.0054597657) q[0];
cx q[6],q[0];
cx q[5],q[0];
cx q[3],q[0];
cx q[2],q[0];
cx q[1],q[0];
s q[9];
h q[9];
cx q[9],q[0];
s q[4];
h q[4];
cx q[4],q[0];
rz(0.021080375) q[0];
cx q[4],q[0];
cx q[9],q[0];
s q[2];
h q[2];
cx q[2],q[9];
s q[4];
h q[4];
cx q[4],q[9];
s q[6];
h q[6];
cx q[6],q[9];
rz(0.0060693137) q[9];
cx q[6],q[9];
h q[6];
sdg q[6];
cx q[4],q[9];
h q[4];
sdg q[4];
cx q[2],q[9];
h q[2];
sdg q[2];
cx q[5],q[6];
cx q[1],q[6];
h q[0];
cx q[0],q[6];
rz(0.047114885) q[6];
cx q[0],q[6];
h q[0];
cx q[1],q[6];
cx q[5],q[6];
cx q[3],q[1];
cx q[4],q[1];
h q[7];
cx q[7],q[1];
cx q[8],q[1];
h q[9];
cx q[9],q[1];
rz(0.0054597657) q[1];
cx q[9],q[1];
h q[9];
cx q[8],q[1];
cx q[7],q[1];
h q[7];
cx q[4],q[1];
cx q[3],q[1];
cx q[8],q[2];
rz(0.13757105) q[2];
cx q[8],q[2];
s q[0];
h q[0];
cx q[0],q[9];
s q[3];
h q[3];
cx q[3],q[9];
cx q[4],q[9];
s q[5];
h q[5];
cx q[5],q[9];
cx q[8],q[9];
rz(-0.009759481) q[9];
cx q[8],q[9];
cx q[5],q[9];
h q[5];
sdg q[5];
cx q[4],q[9];
cx q[3],q[9];
h q[3];
sdg q[3];
cx q[0],q[9];
h q[0];
sdg q[0];
s q[6];
h q[6];
cx q[6],q[8];
s q[2];
h q[2];
cx q[2],q[8];
rz(-0.0023645665) q[8];
cx q[2],q[8];
h q[2];
sdg q[2];
cx q[2],q[0];
rz(0.16720244) q[0];
cx q[2],q[0];
s q[9];
h q[9];
cx q[9],q[0];
cx q[8],q[0];
s q[7];
h q[7];
cx q[7],q[0];
cx q[4],q[0];
cx q[3],q[0];
rz(-0.070233541) q[0];
cx q[3],q[0];
cx q[4],q[0];
cx q[7],q[0];
h q[7];
sdg q[7];
cx q[8],q[0];
cx q[9],q[0];
h q[9];
sdg q[9];
cx q[0],q[1];
rz(0.12491025) q[1];
cx q[0],q[1];
h q[7];
cx q[7],q[8];
s q[6];
h q[6];
cx q[6],q[8];
cx q[5],q[8];
h q[4];
cx q[4],q[8];
cx q[3],q[8];
cx q[1],q[8];
s q[0];
h q[0];
cx q[0],q[8];
rz(0.01315149) q[8];
cx q[0],q[8];
cx q[1],q[8];
cx q[3],q[8];
cx q[4],q[8];
cx q[5],q[8];
cx q[6],q[8];
h q[6];
sdg q[6];
cx q[7],q[8];
h q[3];
cx q[3],q[0];
s q[5];
h q[5];
cx q[5],q[0];
h q[7];
cx q[7],q[0];
rz(0.0097366051) q[0];
cx q[7],q[0];
cx q[5],q[0];
h q[5];
sdg q[5];
cx q[3],q[0];
cx q[9],q[0];
cx q[6],q[0];
cx q[5],q[0];
s q[2];
h q[2];
cx q[2],q[0];
cx q[1],q[0];
rz(-0.066117457) q[0];
cx q[1],q[0];
cx q[2],q[0];
h q[2];
sdg q[2];
cx q[5],q[0];
cx q[6],q[0];
cx q[9],q[0];
s q[1];
h q[1];
cx q[1],q[8];
h q[4];
cx q[4],q[8];
s q[6];
h q[6];
cx q[6],q[8];
rz(0.020657639) q[8];
cx q[6],q[8];
h q[6];
sdg q[6];
cx q[4],q[8];
h q[4];
cx q[1],q[8];
h q[1];
sdg q[1];
cx q[8],q[9];
cx q[6],q[9];
h q[5];
cx q[5],q[9];
cx q[4],q[9];
h q[3];
cx q[3],q[9];
h q[2];
cx q[2],q[9];
cx q[1],q[9];
rz(0.011993522) q[9];
cx q[1],q[9];
cx q[2],q[9];
h q[2];
cx q[3],q[9];
cx q[4],q[9];
cx q[5],q[9];
h q[5];
cx q[6],q[9];
cx q[8],q[9];
s q[2];
h q[2];
cx q[2],q[1];
h q[3];
cx q[3],q[1];
s q[5];
h q[5];
cx q[5],q[1];
cx q[6],q[1];
h q[7];
cx q[7],q[1];
rz(-0.009759481) q[1];
cx q[7],q[1];
h q[7];
cx q[5],q[1];
h q[5];
sdg q[5];
cx q[3],q[1];
h q[9];
cx q[9],q[1];
h q[8];
cx q[8],q[1];
cx q[6],q[1];
cx q[4],q[1];
s q[2];
h q[2];
cx q[2],q[1];
rz(0.011993522) q[1];
cx q[2],q[1];
h q[2];
sdg q[2];
cx q[4],q[1];
cx q[6],q[1];
cx q[8],q[1];
h q[8];
cx q[9],q[1];
h q[9];
h q[0];
cx q[0],q[8];
h q[3];
cx q[3],q[8];
h q[4];
cx q[4],q[8];
h q[5];
cx q[5],q[8];
rz(-0.0069327837) q[8];
cx q[5],q[8];
h q[5];
cx q[4],q[8];
h q[4];
cx q[3],q[8];
h q[3];
cx q[0],q[8];
h q[0];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
h q[1];
h q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
h q[1];
h q[2];
cx q[3],q[6];
cx q[6],q[3];
cx q[3],q[6];
h q[3];
h q[6];
cx q[3],q[6];
cx q[6],q[3];
cx q[3],q[6];
h q[3];
h q[6];
cx q[4],q[8];
cx q[8],q[4];
cx q[4],q[8];
h q[4];
h q[8];
cx q[4],q[8];
cx q[8],q[4];
cx q[4],q[8];
h q[4];
h q[8];
cx q[5],q[1];
cx q[1],q[5];
cx q[5],q[1];
h q[5];
h q[1];
cx q[5],q[1];
cx q[1],q[5];
cx q[5],q[1];
h q[5];
h q[1];
cx q[7],q[5];
cx q[5],q[7];
cx q[7],q[5];
h q[7];
h q[5];
cx q[7],q[5];
cx q[5],q[7];
cx q[7],q[5];
h q[7];
h q[5];
