"""IR&ArF unified training command smoke checks without dataset/GPU access."""
import unittest

from experiment_protocol import build_train_command, resolve_model, protocol_summary


class IRArFLauncherTests(unittest.TestCase):
    def test_default_protocol(self):
        adapter=resolve_model("IR-ArF")
        cmd=build_train_command(adapter,"PaviaU","registered","cuda:0",epochs=15)
        self.assertEqual(cmd[cmd.index("--lr")+1], "0.0002")
        self.assertEqual(cmd[cmd.index("--epochs")+1], "15")
        self.assertEqual(cmd[cmd.index("--stage")+1], "registered")

    def test_explicit_low_lr(self):
        adapter=resolve_model("IR-ArF")
        cmd=build_train_command(adapter,"PaviaU","registered","cuda:0",
                                epochs=15,lr_override=5e-5)
        self.assertEqual(float(cmd[cmd.index("--lr")+1]), 5e-5)
        summary=protocol_summary(adapter,epochs=15,mode="registered",lr_override=5e-5)
        self.assertIn("5e-05",summary)

    def test_other_methods_not_affected(self):
        with self.assertRaisesRegex(ValueError,"IR&ArF"):
            build_train_command(resolve_model("CLSNet"),"PaviaU","registered",
                                "cuda:0",lr_override=5e-5)


if __name__=="__main__":
    unittest.main()
