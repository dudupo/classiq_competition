OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
cx q[0],q[8];
s q[5];
h q[5];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
rz(0.021778815) q[8];
cx q[6],q[8];
cx q[5],q[8];
h q[5];
sdg q[5];
cx q[0],q[8];
cx q[4],q[6];
rz(0.13757105) q[6];
cx q[4],q[6];
h q[5];
cx q[5],q[2];
cx q[6],q[2];
cx q[7],q[2];
h q[8];
cx q[8],q[2];
rz(-0.00070377871) q[2];
cx q[8],q[2];
h q[8];
cx q[7],q[2];
cx q[6],q[2];
cx q[5],q[2];
h q[5];
cx q[9],q[8];
rz(0.10724282) q[8];
cx q[9],q[8];