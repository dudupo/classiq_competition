OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
cx q[1],q[4];
h q[3];
cx q[3],q[4];
rz(0.0072404975) q[4];
h q[3];
cx q[3],q[4];
cx q[1],q[4];
rz(0.0072404975) q[4];
cx q[1],q[4];
cx q[1],q[9];
s q[5];
h q[5];
cx q[5],q[9];
cx q[6],q[9];
cx q[7],q[9];
cx q[8],q[9];
rz(0.0054597657) q[9];
cx q[8],q[9];
cx q[7],q[9];
cx q[6],q[9];
cx q[5],q[9];
h q[5];
sdg q[5];
cx q[1],q[9];
rz(2.1418549) q[9];
cx q[1],q[4];
h q[3];
cx q[3],q[4];
rz(0.0072404975) q[4];
h q[3];
cx q[3],q[4];
cx q[1],q[4];
rz(0.0072404975) q[4];
cx q[1],q[4];
cx q[3],q[4];
h q[0];
cx q[0],q[8];
cx q[1],q[8];
cx q[2],q[8];
h q[3];
cx q[3],q[8];
h q[5];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
rz(0.01315149) q[8];
cx q[7],q[8];
cx q[6],q[8];
h q[5];
cx q[5],q[8];
h q[3];
cx q[3],q[8];
cx q[2],q[8];
cx q[1],q[8];
h q[0];
cx q[0],q[8];
rz(0.01315149) q[8];
cx q[0],q[8];
h q[0];
cx q[2],q[8];
cx q[3],q[8];
h q[3];
cx q[1],q[8];
h q[5];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
rz(-0.00070377871) q[8];
cx q[7],q[8];
cx q[6],q[8];
cx q[5],q[8];
h q[5];
cx q[1],q[8];
s q[4];
h q[4];
cx q[4],q[6];
cx q[3],q[6];
cx q[2],q[6];
cx q[1],q[6];
s q[0];
h q[0];
cx q[0],q[6];
rz(0.0054597657) q[6];
cx q[0],q[6];
h q[0];
sdg q[0];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
h q[4];
sdg q[4];
cx q[1],q[9];
s q[5];
h q[5];
cx q[5],q[9];
cx q[6],q[9];
cx q[7],q[9];
cx q[8],q[9];
rz(0.0054597657) q[9];
cx q[8],q[9];
cx q[7],q[9];
cx q[6],q[9];
cx q[5],q[9];
h q[5];
sdg q[5];
cx q[1],q[9];
rz(2.1418549) q[9];
