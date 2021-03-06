import logo from './logo.svg';
import './App.css';
import React, {useEffect, useState} from 'react';
import axios from 'axios'

function App() {
    const [ setGetMessage] = useState({})
    useEffect(() => {
        axios.get('http://localhost:8090/start').then(response => {
            console.log("SUCCESS", response)
            setGetMessage(response)
        }).catch(error => {
            console.log(error)
        })

    }, [])

    return (
        <div className="App">
            <header className="App-header">
                <img src={logo} className="App-logo" alt="logo"/>
                <p>
                    Edit <code>src/App.js</code> and save to reload.
                </p>
                <a
                    className="App-link"
                    href="https://reactjs.org"
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    Learn React
                </a>
            </header>
        </div>
    );
}

export default App;
