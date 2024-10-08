class Chatbox {
  constructor() {
    this.args = {
      openButton: document.querySelector('.chatbox__button'),
      chatBox: document.querySelector('.chatbox__support'),
      sendButton: document.querySelector('.send__button'),
    }

    this.state = false;
    this.messages = [];
  }

  display() {
    const {openButton, chatBox, sendButton } = this.args;

    if (openButton && chatBox && sendButton) {

      openButton.addEventListener('click', () => this.toggleState(chatBox))
      sendButton.addEventListener('click', () => this.onSendButton(chatBox))

      // Track whether the system is currently processing the function
      let isProcessing = false;

      const node = chatBox.querySelector('input');
      node.addEventListener("keyup", ({key}) => {
        if (key == "Enter" && !isProcessing) {
          isProcessing = true;
          // Call the onSendButton function
          this.onSendButton(chatBox);

          // Simulate asynchronous function (e.g., waiting for response)
          setTimeout(() => {
            isProcessing = false;  // Reset flag after processing is done
          }, 3000); // Example delay, adjust as needed for your function
        }
      })
    } else {
        console.error("One or more required elements are missing.");
    }
  }

  toggleState(chatBox) {
    this.state = !this.state;

    if (this.state) {
      chatBox.classList.add('chatbox--active')
    } else {
      chatBox.classList.remove('chatbox--active')
    }
  }

  onSendButton(chatBox) {
    var textField = chatBox.querySelector('input');
    let text1 = textField.value
    if (text1 === "") {
      return;
    }

    let msg1 = {name: "User", message: text1}
    this.messages.push(msg1)

    // 'http://127.0.0.1:5000/predict'
    fetch($SCRIPT_ROOT + '/predict', {
      method: 'POST',
      body: JSON.stringify({message: text1}),
      mode: 'cors',
      headers: {
        'Content-Type': 'application/json'
      }
    })
    .then(r => r.json())
    .then(r => {
      let msg2 = { name: "iNexus", message: r.answer };
      this.messages.push(msg2);
      this.updateChatText(chatBox);
      textField.value =''
    }).catch(error => {
      console.error('Error: ', error);
      this.updateChatText(chatBox)
      textField.value=''
    });
  }

  updateChatText(chatBox) {
    var html = '';
    this.messages.slice().reverse().forEach(function(item) {
      if (item.name === 'iNexus') {
        html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
      } else {
        html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
      }      
    });
    const chatmessage = chatBox.querySelector('.chatbox__messages');
    chatmessage.innerHTML = html;
  }
}

const chatBox = new Chatbox();
chatBox.display();