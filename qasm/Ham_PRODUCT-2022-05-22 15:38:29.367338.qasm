OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
h q[0];
cx q[0],q[6];
h q[1];
cx q[1],q[6];
h q[2];
cx q[2],q[6];
h q[5];
cx q[5],q[6];
rz(0.12557753) q[6];
cx q[5],q[6];
h q[5];
cx q[2],q[6];
h q[2];
cx q[1],q[6];
h q[1];
cx q[0],q[6];
h q[0];
h q[8];
cx q[8],q[9];
h q[7];
cx q[7],q[9];
h q[4];
cx q[4],q[9];
h q[3];
cx q[3],q[9];
rz(-0.0098351395) q[9];
cx q[3],q[9];
h q[3];
cx q[4],q[9];
h q[4];
cx q[7],q[9];
h q[7];
cx q[8],q[9];
h q[8];
cx q[4],q[3];
cx q[7],q[3];
cx q[8],q[3];
cx q[9],q[3];
rz(-0.0098351395) q[3];
cx q[9],q[3];
cx q[8],q[3];
cx q[7],q[3];
cx q[4],q[3];
cx q[6],q[0];
cx q[5],q[0];
cx q[2],q[0];
cx q[1],q[0];
rz(0.22818327) q[0];
cx q[1],q[0];
cx q[2],q[0];
cx q[5],q[0];
cx q[6],q[0];
cx q[5],q[9];
cx q[6],q[9];
h q[7];
cx q[7],q[9];
h q[8];
cx q[8],q[9];
rz(-0.070233541) q[9];
cx q[8],q[9];
h q[8];
cx q[6],q[9];
cx q[5],q[9];
h q[3];
cx q[3],q[4];
h q[2];
cx q[2],q[4];
h q[1];
cx q[1],q[4];
h q[0];
cx q[0],q[4];
rz(-0.070233541) q[4];
cx q[0],q[4];
h q[0];
cx q[1],q[4];
cx q[2],q[4];
h q[2];
cx q[3],q[4];
h q[3];
cx q[3],q[1];
h q[5];
cx q[5],q[1];
h q[7];
cx q[7],q[1];
h q[9];
cx q[9],q[1];
rz(-0.070233541) q[1];
cx q[9],q[1];
h q[9];
cx q[7],q[1];
h q[7];
cx q[5],q[1];
h q[5];
cx q[3],q[1];
h q[3];
cx q[8],q[0];
h q[6];
cx q[6],q[0];
h q[4];
cx q[4],q[0];
cx q[2],q[0];
rz(-0.070233541) q[0];
cx q[2],q[0];
cx q[4],q[0];
h q[4];
cx q[6],q[0];
h q[6];
cx q[8],q[0];
sxdg q[0];
sxdg q[1];
sxdg q[2];
sxdg q[3];
sxdg q[4];
sxdg q[5];
sxdg q[6];
sxdg q[7];
sxdg q[8];
sxdg q[9];
