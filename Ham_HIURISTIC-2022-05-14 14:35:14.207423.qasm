OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
h q[6];
cx q[6],q[4];
h q[9];
cx q[9],q[4];
rz(0.062788763) q[4];
cx q[9],q[4];
h q[9];
cx q[6],q[4];
s q[9];
h q[9];
cx q[9],q[8];
rz(-0.0049175698) q[8];
cx q[9],q[8];
h q[9];
sdg q[9];
h q[8];
cx q[8],q[9];
rz(-0.0049175698) q[9];
cx q[8],q[9];
h q[8];
cx q[6],q[4];
h q[9];
cx q[9],q[4];
rz(0.062788763) q[4];
cx q[9],q[4];
h q[9];
cx q[6],q[4];
h q[6];
s q[9];
h q[9];
cx q[9],q[8];
rz(-0.0049175698) q[8];
cx q[9],q[8];
h q[9];
sdg q[9];
h q[8];
cx q[8],q[9];
rz(-0.0049175698) q[9];
cx q[8],q[9];
h q[8];
