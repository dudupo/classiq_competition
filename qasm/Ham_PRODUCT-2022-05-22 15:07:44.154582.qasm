OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
s q[1];
h q[1];
cx q[1],q[8];
cx q[3],q[8];
s q[6];
h q[6];
cx q[6],q[8];
h q[7];
cx q[7],q[8];
rz(-0.0069327837) q[8];
cx q[7],q[8];
h q[7];
cx q[6],q[8];
h q[6];
sdg q[6];
cx q[1],q[8];
s q[4];
h q[4];
cx q[4],q[9];
cx q[2],q[9];
rz(0.023467248) q[9];
cx q[2],q[9];
cx q[4],q[9];
cx q[3],q[2];
s q[4];
h q[4];
cx q[4],q[2];
s q[7];
h q[7];
cx q[7],q[2];
cx q[8],q[2];
rz(0.025467828) q[2];
cx q[8],q[2];
cx q[7],q[2];
h q[7];
sdg q[7];
cx q[4],q[2];
h q[4];
sdg q[4];
h q[6];
cx q[6],q[2];
cx q[3],q[2];
rz(-0.0023645665) q[2];
cx q[3],q[2];
cx q[6],q[2];
h q[6];
s q[0];
h q[0];
cx q[0],q[8];
s q[1];
h q[1];
cx q[1],q[8];
h q[4];
cx q[4],q[8];
cx q[5],q[8];
rz(-0.0069327837) q[8];
cx q[5],q[8];
cx q[1],q[8];
h q[1];
sdg q[1];
cx q[0],q[8];
h q[0];
sdg q[0];
h q[7];
cx q[7],q[8];
h q[4];
cx q[4],q[8];
cx q[3],q[8];
cx q[2],q[8];
rz(0.025467828) q[8];
cx q[2],q[8];
cx q[3],q[8];
cx q[4],q[8];
cx q[7],q[8];
h q[7];
cx q[3],q[1];
rz(0.13960362) q[1];
cx q[3],q[1];
cx q[9],q[7];
rz(0.16720244) q[7];
cx q[9],q[7];
cx q[5],q[6];
rz(0.120451) q[6];
cx q[5],q[6];
h q[2];
cx q[2],q[6];
cx q[0],q[6];
rz(0.0055514925) q[6];
cx q[0],q[6];
cx q[2],q[6];
h q[2];
cx q[5],q[4];
h q[9];
cx q[9],q[4];
rz(-0.0023645665) q[4];
cx q[9],q[4];
cx q[5],q[4];
s q[6];
h q[6];
cx q[6],q[2];
cx q[4],q[2];
rz(-0.0030813402) q[2];
cx q[4],q[2];
cx q[6],q[2];
h q[6];
sdg q[6];
s q[0];
h q[0];
cx q[0],q[7];
h q[3];
cx q[3],q[7];
s q[5];
h q[5];
cx q[5],q[7];
rz(0.0097366051) q[7];
cx q[5],q[7];
h q[5];
sdg q[5];
cx q[3],q[7];
h q[3];
cx q[0],q[7];
h q[0];
sdg q[0];
h q[6];
cx q[6],q[9];
h q[4];
cx q[4],q[9];
h q[2];
cx q[2],q[9];
rz(0.0060693137) q[9];
cx q[2],q[9];
h q[2];
cx q[4],q[9];
h q[4];
cx q[6],q[9];
h q[6];
cx q[1],q[0];
rz(0.12491025) q[0];
cx q[1],q[0];
cx q[7],q[2];
rz(0.22867909) q[2];
cx q[7],q[2];
cx q[3],q[9];
cx q[4],q[9];
cx q[6],q[9];
h q[7];
cx q[7],q[9];
cx q[8],q[9];
rz(0.0047358738) q[9];
cx q[8],q[9];
h q[7];
cx q[7],q[9];
cx q[6],q[9];
cx q[5],q[9];
cx q[4],q[9];
cx q[3],q[9];
h q[2];
cx q[2],q[9];
cx q[1],q[9];
h q[0];
cx q[0],q[9];
rz(0.061476654) q[9];
cx q[0],q[9];
h q[0];
cx q[1],q[9];
cx q[2],q[9];
h q[2];
cx q[3],q[9];
cx q[5],q[9];
cx q[7],q[9];
cx q[8],q[9];
s q[2];
h q[2];
cx q[2],q[1];
cx q[4],q[1];
cx q[6],q[1];
h q[8];
cx q[8],q[1];
h q[9];
cx q[9],q[1];
rz(0.011993522) q[1];
cx q[9],q[1];
h q[9];
cx q[8],q[1];
h q[8];
cx q[4],q[1];
h q[7];
cx q[7],q[1];
cx q[6],q[1];
s q[5];
h q[5];
cx q[5],q[1];
h q[3];
cx q[3],q[1];
s q[2];
h q[2];
cx q[2],q[1];
rz(-0.009759481) q[1];
cx q[2],q[1];
h q[2];
sdg q[2];
cx q[3],q[1];
h q[3];
cx q[5],q[1];
h q[5];
sdg q[5];
cx q[6],q[1];
cx q[7],q[1];
h q[7];