OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
cx q[7],q[9];
h q[8];
cx q[8],q[9];
rz(0.0072404975) q[9];
cx q[8],q[9];
h q[8];
cx q[7],q[9];
rz(-0.84896351) q[1];
h q[3];
cx q[3],q[1];
h q[4];
cx q[4],q[1];
rz(0.0072404975) q[1];
cx q[4],q[1];
h q[4];
cx q[3],q[1];
cx q[9],q[1];
rz(0.13757105) q[1];
cx q[9],q[1];
s q[0];
h q[0];
cx q[0],q[9];
cx q[1],q[9];
s q[2];
h q[2];
cx q[2],q[9];
s q[7];
h q[7];
cx q[7],q[9];
cx q[8],q[9];
rz(-0.009759481) q[9];
cx q[7],q[9];
h q[7];
sdg q[7];
cx q[2],q[9];
h q[2];
sdg q[2];
cx q[0],q[9];
h q[0];
sdg q[0];
cx q[8],q[9];
cx q[7],q[9];
cx q[6],q[9];
s q[5];
h q[5];
cx q[5],q[9];
h q[3];
cx q[3],q[9];
cx q[2],q[9];
cx q[1],q[9];
h q[0];
cx q[0],q[9];
rz(-0.015528882) q[9];
cx q[0],q[9];
h q[0];
cx q[1],q[9];
cx q[2],q[9];
cx q[3],q[9];
h q[3];
cx q[5],q[9];
h q[5];
sdg q[5];
cx q[6],q[9];
cx q[7],q[9];
cx q[8],q[9];
