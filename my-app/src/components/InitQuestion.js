import React from "react";

class InitQuestion extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            messages: [],
            isLoading: true
        }
    }

    async init() {
        try {
            const response = await fetch('http://192.148.247.144:8091/init');
            const json = await response.json();
            this.setState({messages: [...this.state.messages, json.first_prompt]});
        } catch (error) {
            console.log(error);
        } finally {
            this.setState({isLoading: false});
        }
    }

    componentDidMount() {
        this.init();
    }

    render() {
        return (
            <div>
                {
                    this.state.messages
                }
            </div>
        )

    }
}

export default InitQuestion;