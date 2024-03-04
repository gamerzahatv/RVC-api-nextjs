import {
  ThemeProvider,
  GlobalStyle,
} from "@react95/core";
import "./global.css"
// import { createGlobalStyle } from 'styled-components'
import Desktop from '../components/desktop';
import ShortcutBox from "../components/desktopBox";




export default function App({ Component, pageProps }) {
  return (
    
    <ThemeProvider>
      <GlobalStyle />
      <ShortcutBox as="main">
      <Desktop>
      <Component {...pageProps} />
      </Desktop>
      </ShortcutBox>
    </ThemeProvider>
  );
}
