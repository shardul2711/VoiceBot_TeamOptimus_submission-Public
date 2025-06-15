import { LoginForm } from "@/components/authForms/login-form";
import AuthLayout from "../layout";
import Image from "next/image";

export default function LoginPage() {
  return (
    <AuthLayout>
      <div className="grid min-h-svh lg:grid-cols-2">
        <div className="flex flex-col gap-4 p-6 md:p-10">
          <div className="flex justify-center gap-2 md:justify-start">
            <div className="flex items-center gap-2 font-medium">
              <Image src="/icons/logo.png" alt="Logo" width={140} height={55} />
            </div>
          </div>
          <div className="flex flex-1 items-center justify-center">
            <div className="w-full max-w-xs">
              <LoginForm />
            </div>
          </div>
        </div>
        <div className="relative hidden bg-muted lg:block">
          <img
            src="/images/LoginPage.png"
            alt="Image"
            className="absolute inset-0 h-full w-full object-cover"
          />
        </div>
      </div>
    </AuthLayout>
  );
}
