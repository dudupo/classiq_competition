OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
cx q[5],q[6];
rz(0.12491025) q[6];
cx q[5],q[6];
cx q[4],q[0];
rz(0.16720244) q[0];
cx q[4],q[0];
cx q[2],q[8];
h q[5];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
rz(-0.00070377871) q[8];
cx q[7],q[8];
cx q[6],q[8];
cx q[5],q[8];
h q[5];
cx q[2],q[8];
s q[6];
h q[6];
cx q[6],q[0];
s q[5];
h q[5];
cx q[5],q[0];
h q[1];
cx q[1],q[0];
rz(0.0097366051) q[0];
cx q[1],q[0];
cx q[5],q[0];
h q[5];
sdg q[5];
cx q[6],q[0];
h q[0];
cx q[0],q[9];
cx q[1],q[9];
cx q[6],q[9];
cx q[7],q[9];
cx q[8],q[9];
rz(-0.009759481) q[9];
cx q[8],q[9];
cx q[7],q[9];
cx q[6],q[9];
h q[6];
sdg q[6];
cx q[1],q[9];
h q[1];
cx q[0],q[9];
h q[0];