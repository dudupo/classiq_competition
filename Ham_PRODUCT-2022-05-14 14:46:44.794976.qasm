OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
h q[3];
cx q[3],q[1];
h q[6];
cx q[6],q[1];
rz(0.062788763) q[1];
cx q[6],q[1];
h q[6];
cx q[3],q[1];
s q[7];
h q[7];
cx q[7],q[8];
rz(-0.0049175698) q[8];
cx q[7],q[8];
h q[7];
sdg q[7];
h q[0];
cx q[0],q[9];
rz(-0.0049175698) q[9];
cx q[0],q[9];
h q[0];
h q[4];
cx q[4],q[5];
h q[9];
cx q[9],q[5];
rz(0.11409164) q[5];
cx q[9],q[5];
cx q[4],q[5];
h q[1];
cx q[1],q[0];
s q[6];
h q[6];
cx q[6],q[0];
h q[7];
cx q[7],q[0];
h q[8];
cx q[8],q[0];
cx q[9],q[0];
rz(-0.03511677) q[0];
cx q[9],q[0];
h q[9];
cx q[8],q[0];
h q[8];
cx q[7],q[0];
h q[7];
cx q[6],q[0];
h q[6];
sdg q[6];
cx q[1],q[0];
h q[1];
cx q[3],q[1];
h q[6];
cx q[6],q[1];
rz(0.062788763) q[1];
cx q[6],q[1];
h q[6];
cx q[3],q[1];
h q[3];
s q[7];
h q[7];
cx q[7],q[8];
rz(-0.0049175698) q[8];
cx q[7],q[8];
h q[7];
sdg q[7];
h q[0];
cx q[0],q[9];
rz(-0.0049175698) q[9];
cx q[0],q[9];
h q[0];
cx q[4],q[5];
h q[9];
cx q[9],q[5];
rz(0.11409164) q[5];
cx q[9],q[5];
cx q[4],q[5];
h q[4];
h q[1];
cx q[1],q[0];
s q[6];
h q[6];
cx q[6],q[0];
h q[7];
cx q[7],q[0];
h q[8];
cx q[8],q[0];
cx q[9],q[0];
rz(-0.03511677) q[0];
cx q[9],q[0];
h q[9];
cx q[8],q[0];
h q[8];
cx q[7],q[0];
h q[7];
cx q[6],q[0];
h q[6];
sdg q[6];
cx q[1],q[0];
h q[1];