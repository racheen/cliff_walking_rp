import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cliff Walking (Positive-Only RL)",
  description: "View-only training visualizations and comparisons."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="container">
          <header className="header">
            <div className="headerInner">
              <div className="title">Cliff Walking (R+)</div>
              <nav className="nav">
                <Link href="/">Home</Link>
                <Link href="/runs">Runs</Link>
                <Link href="/compare">Compare</Link>
                <Link href="/reflection">Reflection</Link>
              </nav>
            </div>
          </header>
          <main className="main">
            <div className="wrap">{children}</div>
          </main>
          <footer className="footer">
            <div className="footerInner">
              <span>View-only artifact browser</span>
              <span className="small">Cliff Walking • traditional vs positive-only</span>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}

